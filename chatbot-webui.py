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
from modules.speech_state import SpeechState
from modules.voice import speak
from modules.thread_manager import set_thread_model, get_thread_model
from modules.thread_manager import get_active_thread, get_thread_history, save_thread_history, list_threads, switch_thread, create_thread, delete_thread
from utils.file_utils import extract_text_from_json
from core import router
import json
import fitz  # PyMuPDF
import os
from datetime import datetime
import threading
from config import WEBUI_TITLE,WEBUI_TOP_PAGE_BANNER,WEBUI_CHATBOT_LABEL,WEBUI_SPEAK_TO_TEXT_LABEL,DEFAULT_LLM_MODEL,CONTEXT_DIR

if SpeechState.get():
    # speech is on
    print("Speech is On")
else:
    # speech is off
    print("Speech is OFF")


#### Remember where we left off
load_context()
speech_state = enabled_speech_default()

#######################################################################################
available_models = [DEFAULT_LLM_MODEL, "mistral-ollama", "mistral", "llama3"]
current_model = gr.State(get_thread_model(get_active_thread()))  # ✅ load model from active thread


def handle_input(user_input, chatbox_display_history, selected_model):
    if chatbox_display_history is None:
        chatbox_display_history = []

    # 🔁 Run the selected model
    response = router.handle_input(user_input, model_name=selected_model)

    chatbox_display_history.append((user_input, response))

    # 🗣️ Async speech
    def async_speak():
        if speech_state:
            speak(response)
    threading.Thread(target=async_speak).start()

    # 🧠 Save history and model
    thread = get_active_thread()
    save_thread_history(thread, chatbox_display_history)
    set_thread_model(thread, selected_model)  # ✅ Save the model for the thread

    return "", chatbox_display_history

############################################################################################

def on_model_change(new_model):
    thread = get_active_thread()
    set_thread_model(thread, new_model)
    return gr.update(value=new_model)

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
    upload_dir = os.path.join(f"{CONTEXT_DIR}/uploads", thread_name)
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
    upload_dir = os.path.join(f"{CONTEXT_DIR}/uploads", thread)
    os.makedirs(upload_dir, exist_ok=True)
    dest_path = os.path.join(upload_dir, os.path.basename(file_path))
    shutil.copy(file_path, dest_path)
    
    # Use filename (without extension) as unique ID for this vector file
    file_id = os.path.splitext(os.path.basename(file_path))[0]

    file_vector_dir = os.path.join(f"{CONTEXT_DIR}/vector_store", thread)
    os.makedirs(file_vector_dir, exist_ok=True)

    faiss_file = os.path.join(file_vector_dir, f"{file_id}.index")
    meta_file = os.path.join(file_vector_dir, f"{file_id}_meta.pkl")

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    faiss.write_index(index, faiss_file)
    with open(meta_file, "wb") as f:
        pickle.dump(chunks, f)



    # 🔹 Optional context summary
    snippet = content[:500].strip()
    if snippet:
        add("Document Upload", f"[Excerpt from {os.path.basename(file_path)}]:\n{snippet}")

    files_md = list_uploaded_files(thread)
    
    return f"📄 Uploaded and indexed `{os.path.basename(file_path)}` for thread `{thread}`.", files_md


################################################################################

def on_switch_thread(selected_name):
    if not selected_name:
        return (
        gr.update(value=history),
        gr.update(value=selected_name),
        gr.update(value=files_md),
        gr.update(value=model)  # ✅ this updates model_selector
    )


    switch_thread(selected_name)
    history = get_thread_history(selected_name) or []
    files_md = list_uploaded_files(selected_name) or "No files uploaded."
    model = get_thread_model(selected_name)  # ✅ Load per-thread model

    return (
        gr.update(value=history),
        gr.update(value=selected_name),
        gr.update(value=files_md),
        gr.update(value=model)  # ✅ update dropdown
    )


def on_new_thread(name=None):
    if not name or name.strip() == "":
        name = f"thread-{datetime.now().strftime('%H%M%S')}"
    create_thread(name)
    switch_thread(name)
    threads = list_threads()
    # Ensure threads is non-empty
    return gr.update(choices=threads or [], value=name), name, []


def on_delete_thread(current):
    import shutil
    import traceback

    try:
        # Delete uploads
        uploads_dir = os.path.join(f"{CONTEXT_DIR}/uploads", current)
        if os.path.exists(uploads_dir):
            shutil.rmtree(uploads_dir)

        # Delete vectors
        vector_dir = os.path.join(f"{CONTEXT_DIR}/vector_store", current)
        if os.path.exists(vector_dir):
            shutil.rmtree(vector_dir)

        # Delete metadata and history
        delete_thread(current)

    except Exception as e:
        traceback.print_exc()
        return gr.update(visible=True, value="❌ Error deleting thread."), current, []

    threads = list_threads()
    if not threads:
        default_thread = "default"
        create_thread(default_thread)
        switch_thread(default_thread)
        threads = list_threads()
        history = get_thread_history(default_thread) or []
        return gr.update(choices=threads, value=default_thread), default_thread, history

    fallback = threads[0]
    switch_thread(fallback)
    history = get_thread_history(fallback) or []
    return gr.update(choices=threads, value=fallback), fallback, history

#################################################################################
import tempfile
import scipy.io.wavfile as wav
from faster_whisper import WhisperModel
# Load the Whisper model once (can be moved to global scope if preferred)
whisper_model = WhisperModel("base", compute_type="int8")

def handle_audio(audio_path, chatbox_display_history):
    if audio_path is None:
        return "", chatbox_display_history

    # Convert audio path to waveform
    try:
        import torchaudio
        waveform, sample_rate = torchaudio.load(audio_path)
        # Save to temp WAV for compatibility
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
            wav.write(temp_wav.name, sample_rate, waveform.squeeze(0).numpy())
            temp_wav_path = temp_wav.name
    except Exception as e:
        return f"❌ Failed to process audio: {e}", chatbox_display_history

    # Transcribe using faster-whisper, then clean up temp file
    try:
        segments, _ = whisper_model.transcribe(temp_wav_path, beam_size=5)
        transcript = " ".join([seg.text.strip() for seg in segments])
    except Exception as e:
        transcript = f"⚠️ Whisper failed: {e}"
    finally:
        # 🔐 Always clean up the temp file
        if os.path.exists(temp_wav_path):
            os.remove(temp_wav_path)


    return handle_input(transcript, chatbox_display_history)

# Ensure there is always a thread
def ensure_default_thread():
    threads = list_threads()
    if not threads:
        create_thread("default")
        switch_thread("default")
        threads = list_threads()
    return threads


############################# GUI ###################################################

threads = ensure_default_thread()

thread_names = gr.State(threads)
active_thread = gr.State(get_active_thread())


with gr.Blocks() as friday_ui:
    gr.HTML("""
    <head>
        <title>Friday - Your AI Assistant</title>
        <meta name="description" content="Friday is your personal AI assistant.">
    </head>
    """)
    gr.Markdown(f"# {WEBUI_TOP_PAGE_BANNER}")
    
    
    with gr.Row():
        thread_selector = gr.Radio(
                choices=list_threads(),           # this returns a list of strings
                label="🧵 Threads",
                value=get_active_thread(),        # this returns the active thread name string
                interactive=True)
        new_thread_name = gr.Textbox(label="➕ New Thread", placeholder="e.g. trip-planning")

    with gr.Row():
        chatbot = gr.Chatbot(label=f"{WEBUI_CHATBOT_LABEL}", value=get_thread_history(get_active_thread()), show_copy_button=True)

    with gr.Row(scale=0.5):
        text_input = gr.Textbox(placeholder="Type here...", scale=4)
        with gr.Column(scale=0.5):
            voice_toggle_btn = gr.Button("[🔈 Audio]" if speech_state else "[🔇 Audio]", scale=0.5)
            send_btn = gr.Button("Send", scale=0.5)
        
    with gr.Accordion("Microphone/Advanced Options", open=False):
        with gr.Row(scale=0.5):
            with gr.Column(scale=0.5):
                model_selector = gr.Dropdown(
                    choices=available_models,
                    value=get_thread_model(get_active_thread()),  # ✅ dynamically assigned
                    label="Model",
                    interactive=True
                    )

            with gr.Column(scale=0.5):
                mic_input = gr.Microphone(label=f"{WEBUI_SPEAK_TO_TEXT_LABEL}", scale=0.5)
    
    # 🔽 (collapsed by default)
    with gr.Accordion("File Management", open=False):    
        with gr.Row(scale=0.5):
            with gr.Column(scale=0.5):
                file_upload = gr.File(label="📁 Upload File (.txt or .pdf)", file_types=[".txt", ".pdf", ".json"], scale=0.5, elem_id="upload_box")
            with gr.Column(scale=0.5):
                
                file_status = gr.Textbox(label="File Status", interactive=False, scale=1)
                uploaded_files_md = gr.Markdown(value=list_uploaded_files(get_active_thread()), label="📁 Uploaded Files")
    # 🔽 (collapsed by default)
    with gr.Accordion("Delete Current Thread", open=False):
                delete_thread_btn = gr.Button("🗑️ Delete Current Thread")


# Bind events AFTER creating the widgets
    send_btn.click(handle_input, inputs=[text_input, chatbot, model_selector], outputs=[text_input, chatbot])
    text_input.submit(handle_input, inputs=[text_input, chatbot, model_selector], outputs=[text_input, chatbot])
    file_upload.change(handle_file, inputs=file_upload, outputs=[file_status, uploaded_files_md])       

    voice_toggle_btn.click(toggle_voice, [], [voice_toggle_btn])
   
    mic_input.change(handle_audio, [mic_input, chatbot], outputs=[text_input, chatbot])
    
    model_selector.change(on_model_change, inputs=[model_selector], outputs=[model_selector])
    thread_selector.change(on_switch_thread, inputs=[thread_selector], outputs=[chatbot, active_thread, uploaded_files_md, model_selector])
    delete_thread_btn.click(on_delete_thread, inputs=[thread_selector], outputs=[thread_selector, active_thread, chatbot])
    
    new_thread_name.submit(on_new_thread, inputs=[new_thread_name], outputs=[thread_selector, thread_selector, chatbot])

friday_ui.launch()
