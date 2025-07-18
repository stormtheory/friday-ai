# Written by StormTheory
# Ported from Gradio frontend to customtkinter

import customtkinter as ctk
from tkinter import filedialog, messagebox, simpledialog
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
import pathlib
from utils.file_utils import extract_text_from_json
from config import DEFAULT_LLM_MODEL, CHATBOT_TITLE, ASSISTANT_PROMPT_NAME, USER_PROMPT_NAME, CONTEXT_DIR
from core import router

# Load context once at app start
load_context()

# Shared components
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=400, separators=["\n\n", "\n", ".", "!", "?"])
whisper_model = WhisperModel("base", compute_type="int8")


def list_uploaded_files(thread_name):
    upload_dir = os.path.join(f"{CONTEXT_DIR}/uploads", thread_name)
    if not os.path.exists(upload_dir):
        return "No files uploaded."
    files = os.listdir(upload_dir)
    if not files:
        return "No files uploaded."
    lines = [f"- {f} ({os.path.getsize(os.path.join(upload_dir, f)) / 1024:.1f} KB)"
             for f in files]
    return "\n".join(lines)


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

        # Thinking indicator label
        self.thinking_label = ctk.CTkLabel(self, text="Status: Idle", text_color="gray")
        self.thinking_label.pack(pady=5)

        # Frame to hold grouped buttons horizontally and center them
        self.button_row = ctk.CTkFrame(self)
        self.button_row.pack(pady=10, anchor="center")

        # Buttons packed side-by-side inside the row
        self.send_button = ctk.CTkButton(self.button_row, text="Send", command=self.on_send)
        self.send_button.pack(side="left", padx=5)

        self.voice_btn = ctk.CTkButton(self.button_row, text="🔈 Audio On" if self.speech_enabled else "🔇 Audio Off", command=self.toggle_voice)
        self.voice_btn.pack(side="left", padx=5)

        self.upload_btn = ctk.CTkButton(self.button_row, text="📁 Upload File (.txt/.pdf/.json)", command=self.select_and_process_file)
        self.upload_btn.pack(side="left", padx=5)

        self.new_thread_btn = ctk.CTkButton(self.button_row, text="➕ New Thread", command=self.create_new_thread)
        self.new_thread_btn.pack(side="left", padx=5)

        self.delete_thread_btn = ctk.CTkButton(self.button_row, text="🗑️ Delete Thread", command=self.delete_current_thread)
        self.delete_thread_btn.pack(side="left", padx=5)

        # --- Uploaded Files Label & Viewer ---
        self.files_label = ctk.CTkLabel(self, text="📁 Uploaded Files:", font=ctk.CTkFont(size=14, weight="bold"))
        self.files_label.pack(pady=(10, 0))

        # Create a container frame with fixed height
        self.files_container = ctk.CTkFrame(self, width=850, height=120)
        self.files_container.pack(pady=5)
        self.files_container.pack_propagate(False)  # Prevent auto-resizing

        # Inside it, add the scrollable frame (will scroll if content exceeds 120px)
        self.files_frame = ctk.CTkScrollableFrame(self.files_container, width=850)
        self.files_frame.pack(fill="both", expand=True)


        # Scrollable frame to hold file entries and delete buttons
        #self.files_frame = ctk.CTkScrollableFrame(self, width=850, height=50)
        #self.files_frame.pack(pady=5)
        
        #self.files_box = ctk.CTkTextbox(self, width=850, height=80)
        #self.files_box.configure(state="disabled")
        #self.files_box.pack(pady=5)
        # --------------------------------------

        self.refresh_files_display()

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

    def refresh_files_display(self):
        # Clear old widgets in scrollable frame
        for widget in self.files_frame.winfo_children():
            widget.destroy()
        upload_dir = os.path.join(f"{CONTEXT_DIR}/uploads", self.active_thread)
        if not os.path.exists(upload_dir):
            return
        files = os.listdir(upload_dir)
        if not files:
            no_files_label = ctk.CTkLabel(self.files_frame, text="No files uploaded.")
            no_files_label.pack(anchor="w")
            return
        for filename in files:
            file_path = os.path.join(upload_dir, filename)
            size_kb = os.path.getsize(file_path) / 1024

            file_row = ctk.CTkFrame(self.files_frame)
            file_row.pack(fill="x", pady=2)

            label = ctk.CTkLabel(file_row, text=f"{filename} ({size_kb:.1f} KB)", anchor="w", width=700)
            label.pack(side="left", padx=5)

            del_btn = ctk.CTkButton(file_row, text="❌ Delete", width=80,
                                    command=lambda f=filename: self.delete_uploaded_file(f))
            del_btn.pack(side="right", padx=5)

    def delete_uploaded_file(self, filename):
        upload_dir = os.path.join(f"{CONTEXT_DIR}/uploads", self.active_thread)
        file_path = os.path.join(upload_dir, filename)
        # Confirm deletion for safety
        confirm = messagebox.askyesno(
            title="Delete File",
            message=f"Are you sure you want to delete '{filename}' and its embeddings?"
        )
        if not confirm:
            return
        try:
            os.remove(file_path)
            # Remove associated vector files
            file_id = os.path.splitext(filename)[0]
            vector_dir = os.path.join(f"{CONTEXT_DIR}/vector_store", self.active_thread)
            faiss_file = os.path.join(vector_dir, f"{file_id}.index")
            meta_file = os.path.join(vector_dir, f"{file_id}_meta.pkl")
            if os.path.exists(faiss_file):
                os.remove(faiss_file)
            if os.path.exists(meta_file):
                os.remove(meta_file)
            self.refresh_files_display()
        except Exception as e:
            messagebox.showerror("Error", f"❌ Failed to delete file or vectors: {e}")


    def create_new_thread(self):
        new_thread_name = simpledialog.askstring("New Thread", "Enter thread name:")
        if not new_thread_name:
            return
        if new_thread_name in list_threads():
            messagebox.showinfo("Thread Exists", "⚠️ A thread with that name already exists.")
            return

        # Create the thread and switch to it
        create_thread(new_thread_name)
        switch_thread(new_thread_name)
        self.active_thread = new_thread_name
        self.model = get_thread_model(new_thread_name)
        self.chat_history = []
        
        # Update thread selector options
        self.thread_selector.configure(values=list_threads())
        self.thread_selector.set(new_thread_name)
        self.refresh_chat_display()
        self.refresh_files_display()

    def on_send(self):
        user_input = self.input_field.get().strip()
        if not user_input:
            return
        self.input_field.delete(0, 'end')

        # Start response processing in a background thread to avoid UI freeze
        threading.Thread(target=self.process_input, args=(user_input,), daemon=True).start()

    def process_input(self, user_input):
        self.thinking_label.configure(text="🤔 Thinking...", text_color="orange")
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

        self.thinking_label.configure(text="Status: Idle", text_color="gray")

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
        self.refresh_files_display()

    def select_and_process_file(self):
        downloads_path = os.path.join(pathlib.Path.home(), "Downloads")
        file_path = filedialog.askopenfilename(
            initialdir=downloads_path,
            filetypes=[("Supported", "*.txt *.pdf *.json")]
        )

        if file_path:
            status, _ = self.handle_file(file_path)
            messagebox.showinfo(title="Upload", message=status)
            self.refresh_files_display()

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
        upload_dir = os.path.join(f"{CONTEXT_DIR}/uploads", thread)
        os.makedirs(upload_dir, exist_ok=True)
        dest_path = os.path.join(upload_dir, os.path.basename(file_path))
        shutil.copy(file_path, dest_path)

        # Save vector index and metadata
        # Use hashed file name or full file name for ID
        file_id = os.path.splitext(os.path.basename(file_path))[0]

        # Save vectors and metadata per file
        file_vector_dir = os.path.join(f"{CONTEXT_DIR}/vector_store/{self.active_thread}")
        os.makedirs(file_vector_dir, exist_ok=True)

        faiss_file = os.path.join(file_vector_dir, f"{file_id}.index")
        meta_file = os.path.join(file_vector_dir, f"{file_id}_meta.pkl")

        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)
        faiss.write_index(index, faiss_file)

        with open(meta_file, "wb") as f:
            pickle.dump(chunks, f)


        snippet = content[:500].strip()
        if snippet:
            add("Document Upload", f"[Excerpt from {os.path.basename(file_path)}]:\n{snippet}")

        return f"📄 Uploaded and indexed `{os.path.basename(file_path)}`", None

    import traceback    # <- handy for debugging; remove if you don’t want it

    def delete_current_thread(self):
        thread = self.active_thread

        # Single confirmation – message lists everything that will be removed
        confirm = messagebox.askyesno(
            title="Delete Thread",
            message=(
                f"⚠️ This will permanently delete the thread '{thread}',\n"
                f"all chat history, uploaded files, and embeddings.\n"
                "Do you want to continue?"
            )
        )
        if not confirm:
            return

        try:
            # 1️⃣  Remove uploads directory (if it exists)
            uploads_dir = os.path.join(f"{CONTEXT_DIR}/uploads", thread)
            if os.path.exists(uploads_dir):
                shutil.rmtree(uploads_dir)

            # 2️⃣  Remove vector store directory (if it exists)
            vector_dir = os.path.join(f"{CONTEXT_DIR}/vector_store", thread)
            if os.path.exists(vector_dir):
                shutil.rmtree(vector_dir)

            # 3️⃣  Delete thread metadata / history (your helper)
            delete_thread(thread)

        except Exception as e:
            traceback.print_exc()        # optional: log full trace
            messagebox.showerror("Error", f"❌ Failed to delete thread:\n{e}")
            return   # abort UI refresh – keep state unchanged

        # 4️⃣  Pick a fallback thread (or create a fresh 'default')
        threads = list_threads()

        # Fallback if no threads left
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
