# Written by StormTheory
# https://github.com/stormtheory/friday-ai

from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
import faiss
import pickle
import numpy as np

from modules.context import add,load_context

import gradio as gr
from modules.voice import enabled_speech_default,is_speech_enabled,stop_audio
from modules.speech_state import speech_state
from modules.voice import speak  # Your gTTS or pyttsx3 wrapper
from modules.thread_manager import get_active_thread, get_thread_history, save_thread_history, list_threads, switch_thread, create_thread, delete_thread
from utils.file_utils import extract_text_from_json
import json
import fitz  # PyMuPDF
import os
from datetime import datetime
import threading

if speech_state:
    print("Speech is ON")
else:
    print("Speech is OFF")


#### Remember where we left off
load_context()
 
#######################################################################################
def handle_input(user_input, chatbox_display_history):
    if chatbox_display_history is None:
        chatbox_display_history = []

    from core import router  # Ensure import at the top
    # 🔁 Run the model
    response = router.handle_input(user_input)

    chatbox_display_history.append((user_input, response))

    def async_speak():
        if speech_state:
            speak(response)
    threading.Thread(target=async_speak).start()

    # Save chat thread history
    thread = get_active_thread()
    save_thread_history(thread, chatbox_display_history)

    return "", chatbox_display_history


############################################################################################

def toggle_voice():
    # Toggle the global state from speech_state module
    global speech_state
    speech_state = not speech_state

    if speech_state:
        print("🔈 Speech enabled")
        return "🔈 Speech enabled"
    else:
        stop_audio()
        print("🔇 Speech disabled")
        return "🔇 Speech disabled"

#############################################################################

def list_uploaded_files(thread_name):
    upload_dir = os.path.join("uploads", thread_name)
    if not os.path.exists(upload_dir):
        return "No files uploaded for this thread."

    files = os.listdir(upload_dir)
    if not files:
        return "No files uploaded for this thread."

    file_list_md = "### Uploaded files:\n"
    for f in files:
        full_path = os.path.join(upload_dir, f)
        size_kb = os.path.getsize(full_path) / 1024
        file_list_md += f"- {f} ({size_kb:.1f} KB)\n"
    return file_list_md



###############################################################################

# Load once globally
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=400, separators=["\n\n", "\n", ".", "!", "?"])

def handle_file(file_path):
    if not file_path:
        return "⚠️ No file selected."
    
    import shutil

    ext = os.path.splitext(file_path)[1].lower()
    content = ""

    try:
        if ext == ".txt":
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

        elif ext == ".pdf":
            doc = fitz.open(file_path)
            content = "".join([page.get_text() for page in doc])
            doc.close()

        elif ext == ".json":
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            content = extract_text_from_json(data)

        else:
            return "❌ Unsupported file type. Only .txt, .pdf, and .json are allowed."

    except Exception as e:
        return f"❌ Error reading file: {e}"

    # 🔹 Chunk + embed
    chunks = splitter.split_text(content)
    if not chunks:
        return "❌ No valid chunks generated from the file."

    embeddings = embed_model.encode(chunks)
    embeddings = np.array(embeddings)

    if embeddings.ndim != 2 or embeddings.shape[0] == 0:
        return "❌ Failed to generate valid embeddings."

    # 🔹 Save vector index + metadata
    thread = get_active_thread()
    
    # Save file to uploads/{thread}/ folder
    upload_dir = os.path.join("uploads", thread)
    os.makedirs(upload_dir, exist_ok=True)
    dest_path = os.path.join(upload_dir, os.path.basename(file_path))
    shutil.copy(file_path, dest_path)
    
    index_path = f"vector_store/{thread}.index"
    meta_path = f"vector_store/{thread}_meta.pkl"

    os.makedirs("vector_store", exist_ok=True)
    index = faiss.IndexFlatL2(embeddings.shape[1])  # Correct dimension
    index.add(embeddings)

    faiss.write_index(index, index_path)

    with open(meta_path, "wb") as f:
        pickle.dump(chunks, f)

    # 🔹 Optional context summary
    snippet = content[:500].strip()
    if snippet:
        add("Document Upload", f"[Excerpt from {os.path.basename(file_path)}]:\n{snippet}")

    files_md = list_uploaded_files(thread)
    
    return f"📄 Uploaded and indexed `{os.path.basename(file_path)}` for thread `{thread}`.", files_md


################################################################################

def on_switch_thread(selected_name):
    switch_thread(selected_name)
    history = get_thread_history(selected_name)
    files_md = list_uploaded_files(selected_name)
    return gr.update(value=history), gr.update(value=selected_name), gr.update(value=files_md)

def on_new_thread(name=None):
    if not name or name.strip() == "":
        name = f"thread-{datetime.now().strftime('%H%M%S')}"
    create_thread(name)
    switch_thread(name)
    threads = list_threads()
    return gr.update(choices=threads, value=name), name, []

def on_delete_thread(current):
    delete_thread(current)
    threads = list_threads()
    fallback = threads[0] if threads else None
    if fallback:
        switch_thread(fallback)
        history = get_thread_history(fallback)
    else:
        history = []
    return gr.update(choices=threads, value=fallback), fallback, history

#################################################################################

def handle_audio(audio_path, chatbox_display_history):
    if audio_path is None:
        return "", chatbox_display_history
    import speech_recognition as sr
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio = recognizer.record(source)
    try:
        transcript = recognizer.recognize_google(audio)
    except Exception:
        transcript = "⚠️ Could not understand audio."
    return handle_input(transcript, chatbox_display_history)


############################# GUI ###################################################

thread_names = gr.State(list_threads())
active_thread = gr.State(get_active_thread())

with gr.Blocks() as friday_ui:
    gr.HTML("""
    <head>
        <title>Friday - Your AI Assistant</title>
        <meta name="description" content="Friday is your personal AI assistant.">
    </head>
    """)
    gr.Markdown("# 🤖 Friday — Your Local AI Assistant")
    
    
    with gr.Row():
        with gr.Column(scale=1):
            thread_selector = gr.Radio(
                choices=list_threads(),           # this returns a list of strings
                label="🧵 Threads",
                value=get_active_thread(),        # this returns the active thread name string
                interactive=True
)
    with gr.Row():
        chatbot = gr.Chatbot(label="Friday", value=get_thread_history(get_active_thread()), show_copy_button=True)

    with gr.Row():
        text_input = gr.Textbox(placeholder="Type here...", scale=4)

        with gr.Column(scale=1):
            speech_state = enabled_speech_default()
            voice_toggle_btn = gr.Button("[🔈 Speech enabled]" if speech_state else "[🔇 Speech disabled]")
            send_btn = gr.Button("Send")
            
        with gr.Column(scale=1):
            mic_input = gr.Microphone(label="🎤 Speak to Friday", scale=1)
        
    with gr.Row():
        with gr.Column(scale=1):
            file_upload = gr.File(label="📁 Upload File (.txt or .pdf)", file_types=[".txt", ".pdf", ".json"], scale=0.5, elem_id="upload_box")
        with gr.Column(scale=1):
            new_thread_name = gr.Textbox(label="➕ New Thread", placeholder="e.g. trip-planning")
            file_status = gr.Textbox(label="File Status", interactive=False, scale=1)
            uploaded_files_md = gr.Markdown(value=list_uploaded_files(get_active_thread()), label="📁 Uploaded Files")
            delete_thread_btn = gr.Button("🗑️ Delete Current Thread")


# Bind events AFTER creating the widgets
    send_btn.click(handle_input, inputs=[text_input, chatbot], outputs=[text_input, chatbot])
    text_input.submit(handle_input, inputs=[text_input, chatbot], outputs=[text_input, chatbot])
    file_upload.change(handle_file, inputs=file_upload, outputs=[file_status, uploaded_files_md])       

    voice_toggle_btn.click(toggle_voice, [], [voice_toggle_btn])
   
    mic_input.change(handle_audio, [mic_input, chatbot], outputs=[text_input, chatbot])

    thread_selector.change(on_switch_thread, inputs=[thread_selector], outputs=[chatbot, active_thread, uploaded_files_md])
    delete_thread_btn.click(on_delete_thread, inputs=[thread_selector], outputs=[thread_selector, active_thread, chatbot])
    
    new_thread_name.submit(on_new_thread, inputs=[new_thread_name], outputs=[thread_selector, thread_selector, chatbot])


friday_ui.launch()
