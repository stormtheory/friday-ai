# Written by StormTheory
# Ported from Gradio frontend to customtkinter

import customtkinter as ctk
import threading
import os
import json
import shutil
import numpy as np
import pickle

from faster_whisper import WhisperModel
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
import faiss

from modules.context import add, load_context
from modules.voice import speak, stop_audio, enabled_speech_default
from modules.thread_manager import (
    set_thread_model, get_thread_model,
    get_active_thread, get_thread_history, save_thread_history,
    list_threads, switch_thread, create_thread, delete_thread
)
from utils.file_utils import extract_text_from_json
from config import DEFAULT_LLM_MODEL,CHATBOT_TITLE,ASSISTANT_PROMPT_NAME,USER_PROMPT_NAME
from core import router

# Load context once at app start
load_context()

# Shared components
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=400, separators=["\n\n", "\n", ".", "!", "?"])
whisper_model = WhisperModel("base", compute_type="int8")


class FridayApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title(CHATBOT_TITLE)
        self.geometry("960x720")

        self.speech_enabled = enabled_speech_default()
        self.active_thread = get_active_thread()
        self.chat_history = get_thread_history(self.active_thread) or []
        self.model = get_thread_model(self.active_thread)

        self.build_ui()

    def build_ui(self):
        ctk.set_appearance_mode("System")
        ctk.set_default_color_theme("blue")

        self.thread_selector = ctk.CTkOptionMenu(
            self, values=list_threads(), command=self.on_switch_thread
        )
        self.thread_selector.set(self.active_thread)
        self.thread_selector.pack(pady=10)

        self.chatbox = ctk.CTkTextbox(self, width=850, height=400)
        self.chatbox.pack(pady=5)
        self.refresh_chat_display()

        self.input_field = ctk.CTkEntry(self, placeholder_text="Type your message...", width=600)
        self.input_field.pack(pady=10)
        self.input_field.bind("<Return>", self.on_enter_pressed)

        # Thinking indicator label (hidden by default)
        self.thinking_label = ctk.CTkLabel(self, text="🤔 Thinking...", text_color="gray")
        self.thinking_label.pack(pady=5)
        self.thinking_label.pack_forget()

        self.send_button = ctk.CTkButton(self, text="Send", command=self.on_send)
        self.send_button.pack(pady=5)

        self.voice_btn = ctk.CTkButton(self, text="🔈 Audio On" if self.speech_enabled else "🔇 Audio Off", command=self.toggle_voice)
        self.voice_btn.pack(pady=5)

        self.upload_btn = ctk.CTkButton(self, text="📁 Upload File (.txt/.pdf/.json)", command=self.select_and_process_file)
        self.upload_btn.pack(pady=5)

        self.delete_thread_btn = ctk.CTkButton(self, text="🗑️ Delete Thread", command=self.delete_current_thread)
        self.delete_thread_btn.pack(pady=10)

    def refresh_chat_display(self):
        self.chatbox.delete("1.0", "end")
        # Configure a tag for user's text with color white
        self.chatbox.tag_config("user_text", foreground="#FFD700")
        
        # Configure a tag for assistant's text with color yellow
        self.chatbox.tag_config("assistant_text", foreground="#FFFFFF")
        
        for user, response in self.chat_history:
            # Insert user message with 'user_text' tag for white color
            self.chatbox.insert("end", f"🧑 {USER_PROMPT_NAME}: {user}\n", "user_text")
            
            # Insert assistant response with 'assistant_text' tag for yellow color
            self.chatbox.insert("end", f"🤖 {ASSISTANT_PROMPT_NAME}: {response}\n\n", "assistant_text")
        
        # Scroll to the bottom so latest messages are visible
        self.chatbox.see("end")



    def on_send(self):
        user_input = self.input_field.get().strip()
        if not user_input:
            return
        self.input_field.delete(0, 'end')

        # Start response processing in a background thread to avoid UI freeze
        threading.Thread(target=self.process_input, args=(user_input,), daemon=True).start()

    def process_input(self, user_input):
        self.thinking_label.pack()  # Show "Thinking..." indicator
        self.update_idletasks()

        try:
            response = router.handle_input(user_input, model_name=self.model)
        except Exception as e:
            response = f"⚠️ Error: {e}"

        self.chat_history.append((user_input, response))
        self.refresh_chat_display()
        save_thread_history(self.active_thread, self.chat_history)
        set_thread_model(self.active_thread, self.model)

        if self.speech_enabled:
            threading.Thread(target=speak, args=(response,), daemon=True).start()

        self.thinking_label.pack_forget()  # Hide "Thinking..." indicator

    def toggle_voice(self):
        self.speech_enabled = not self.speech_enabled
        if self.speech_enabled:
            self.voice_btn.configure(text="🔈 Audio On")
        else:
            stop_audio()
            self.voice_btn.configure(text="🔇 Audio Off")

    def on_switch_thread(self, thread_name):
        switch_thread(thread_name)
        self.active_thread = thread_name
        self.model = get_thread_model(thread_name)
        self.chat_history = get_thread_history(thread_name) or []
        self.refresh_chat_display()

    def select_and_process_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("Supported", "*.txt *.pdf *.json")])
        if file_path:
            status, _ = self.handle_file(file_path)
            ctk.CTkMessagebox(title="Upload", message=status)

    def on_enter_pressed(self, event):
        self.on_send()

    def handle_file(self, file_path):
        ext = os.path.splitext(file_path)[1].lower()
        content = ""
        try:
            if ext == ".txt":
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
            elif ext == ".pdf":
                import fitz
                doc = fitz.open(file_path)
                content = "".join([page.get_text() for page in doc])
                doc.close()
            elif ext == ".json":
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                content = extract_text_from_json(data)
            else:
                return "❌ Unsupported file type.", None
        except Exception as e:
            return f"❌ Error reading file: {e}", None

        # Text chunking
        chunks = splitter.split_text(content)
        if not chunks:
            return "❌ No valid text chunks extracted.", None

        embeddings = embed_model.encode(chunks)
        embeddings = np.array(embeddings)
        if embeddings.ndim != 2 or embeddings.shape[0] == 0:
            return "❌ Invalid embeddings generated.", None

        # Store file in current thread's upload directory
        thread = self.active_thread
        upload_dir = os.path.join("uploads", thread)
        os.makedirs(upload_dir, exist_ok=True)
        dest_path = os.path.join(upload_dir, os.path.basename(file_path))
        shutil.copy(file_path, dest_path)

        # Save vector index and metadata
        index_path = f"vector_store/{thread}.index"
        meta_path = f"vector_store/{thread}_meta.pkl"
        os.makedirs("vector_store", exist_ok=True)

        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)
        faiss.write_index(index, index_path)
        with open(meta_path, "wb") as f:
            pickle.dump(chunks, f)

        snippet = content[:500].strip()
        if snippet:
            add("Document Upload", f"[Excerpt from {os.path.basename(file_path)}]:\n{snippet}")

        return f"📄 Uploaded and indexed `{os.path.basename(file_path)}`", None

    def delete_current_thread(self):
        delete_thread(self.active_thread)
        threads = list_threads()
        if not threads:
            create_thread("default")
            switch_thread("default")
            threads = list_threads()

        fallback = threads[0]
        self.thread_selector.configure(values=threads)
        self.thread_selector.set(fallback)
        self.on_switch_thread(fallback)


if __name__ == "__main__":
    app = FridayApp()
    app.mainloop()
