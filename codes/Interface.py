import tkinter as tk
from tkinter import ttk, filedialog, messagebox, Canvas
from PIL import Image, ImageTk
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from diarNS import run_diarization
from mypredict_imp import predict_speaker_count
import wave
import contextlib
from tkinter import font as tkfont

class ModernInterface:
    def __init__(self, root):
        self.root = root
        self.root.title("Blind Speaker Diarization")
        self.root.geometry("1200x800")
        self.root.configure(bg="#1A1A2E")
        
        try:
            self.root.iconbitmap("microphone_icon.ico")  
        except:
            pass
        
        self.title_font = tkfont.Font(family="Helvetica", size=18, weight="bold")
        self.subtitle_font = tkfont.Font(family="Helvetica", size=12)
        self.button_font = tkfont.Font(family="Helvetica", size=10, weight="bold")
        
        self.setup_ui()

    # --- Helper function to convert matplotlib RGBA color to Tkinter-compatible hex string ---
    def rgba_to_hex(self, rgba):
        r, g, b, a = rgba
        return f'#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}'
    
    def setup_ui(self):
        self.main_frame = tk.Frame(self.root, bg="#1A1A2E")
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        self.title_frame = tk.Frame(self.main_frame, bg="#16213E")
        self.title_frame.pack(fill=tk.X, pady=(0, 20))
        
        self.title_label = tk.Label(
            self.title_frame, 
            text="Blind Speaker Diarization", 
            font=self.title_font,
            fg="#00F5FF",
            bg="#16213E"
        )
        self.title_label.pack(side=tk.LEFT, padx=20, pady=10)
        
        self.about_btn = tk.Button(
            self.title_frame,
            text="About",
            font=self.button_font,
            bg="#9457EB",
            fg="white",
            bd=0,
            activebackground="#B36BFF",
            activeforeground="white",
            command=self.show_about
        )
        self.about_btn.pack(side=tk.RIGHT, padx=20, pady=10)
        
        self.file_frame = tk.Frame(self.main_frame, bg="#16213E", bd=2, relief=tk.GROOVE)
        self.file_frame.pack(fill=tk.X, pady=(0, 20))
        
        tk.Label(
            self.file_frame,
            text="Upload Audio File",
            font=self.subtitle_font,
            fg="white",
            bg="#16213E"
        ).pack(pady=(10, 5))
        
        self.browse_btn = tk.Button(
            self.file_frame,
            text="Browse Files",
            font=self.button_font,
            bg="#00F5FF",
            fg="#1A1A2E",
            bd=0,
            activebackground="#4DFFFF",
            activeforeground="#1A1A2E",
            command=self.browse_file
        )
        self.browse_btn.pack(pady=(0, 10))
        
        self.file_info_frame = tk.Frame(self.file_frame, bg="#16213E")
        self.file_info_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.file_name_label = tk.Label(
            self.file_info_frame,
            text="No file selected",
            font=self.subtitle_font,
            fg="#AAAAAA",
            bg="#16213E",
            anchor=tk.W
        )
        self.file_name_label.pack(fill=tk.X, padx=20)
        
        self.file_duration_label = tk.Label(
            self.file_info_frame,
            text="Duration: --:--",
            font=self.subtitle_font,
            fg="#AAAAAA",
            bg="#16213E",
            anchor=tk.W
        )
        self.file_duration_label.pack(fill=tk.X, padx=20)
        
        self.control_frame = tk.Frame(self.main_frame, bg="#16213E", bd=2, relief=tk.GROOVE)
        self.control_frame.pack(fill=tk.X, pady=(0, 20))
        
        self.control_buttons_frame = tk.Frame(self.control_frame, bg="#16213E")
        self.control_buttons_frame.pack(pady=10)
        
        self.play_btn = tk.Button(
            self.control_buttons_frame,
            text="▶",
            font=("Helvetica", 14),
            bg="#00F5FF",
            fg="#1A1A2E",
            bd=0,
            width=3,
            state=tk.DISABLED,
            command=self.play_audio
        )
        self.play_btn.pack(side=tk.LEFT, padx=5)
        
        self.stop_btn = tk.Button(
            self.control_buttons_frame,
            text="■",
            font=("Helvetica", 14),
            bg="#9457EB",
            fg="white",
            bd=0,
            width=3,
            state=tk.DISABLED,
            command=self.stop_audio
        )
        self.stop_btn.pack(side=tk.LEFT, padx=5)
        
        self.progress_frame = tk.Frame(self.control_frame, bg="#16213E")
        self.progress_frame.pack(fill=tk.X, padx=20, pady=(0, 10))
        
        self.progress_label = tk.Label(
            self.progress_frame,
            text="00:00 / 00:00",
            font=self.subtitle_font,
            fg="white",
            bg="#16213E",
            width=10
        )
        self.progress_label.pack(side=tk.RIGHT)
        
        self.progress_bar = ttk.Progressbar(
            self.progress_frame,
            orient=tk.HORIZONTAL,
            mode='determinate',
            length=600
        )
        self.progress_bar.pack(fill=tk.X, expand=True, side=tk.LEFT)
        
        self.results_frame = tk.Frame(self.main_frame, bg="#16213E", bd=2, relief=tk.GROOVE)
        self.results_frame.pack(fill=tk.BOTH, expand=True)
        
        tk.Label(
            self.results_frame,
            text="Results",
            font=self.subtitle_font,
            fg="white",
            bg="#16213E"
        ).pack(pady=(10, 5))
        
        self.results_notebook = ttk.Notebook(self.results_frame)
        self.results_notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        
        # Tabs
        self.stats_tab = tk.Frame(self.results_notebook, bg="#1A1A2E")
        self.results_notebook.add(self.stats_tab, text="Speaker Statistics")
        
        self.visualization_tab = tk.Frame(self.results_notebook, bg="#1A1A2E")
        self.results_notebook.add(self.visualization_tab, text="Visualization")
        
        self.output_tab = tk.Frame(self.results_notebook, bg="#1A1A2E")
        self.results_notebook.add(self.output_tab, text="Output Files")
        
        # Variables initialization
        self.audio_file = None
        self.playing = False
        self.speaker_stats = None
        
    def browse_file(self):
        file_path = filedialog.askopenfilename(
            title="Select an audio file",
            filetypes=[("Audio Files", "*.wav *.flac *.mp3 *.ogg")]
        )
        
        if file_path:
            self.audio_file = file_path
            self.file_name_label.config(text=f"File: {os.path.basename(file_path)}")
            
            # Calculate audio duration
            try:
                with contextlib.closing(wave.open(file_path, 'r')) as f:
                    frames = f.getnframes()
                    rate = f.getframerate()
                    duration = frames / float(rate)
                    mins, secs = divmod(duration, 60)
                    self.file_duration_label.config(text=f"Duration: {int(mins):02d}:{int(secs):02d}")
            except:
                self.file_duration_label.config(text="Duration: Unknown")
            
            self.play_btn.config(state=tk.NORMAL)
            self.stop_btn.config(state=tk.NORMAL)
            
            # Process audio file
            self.process_audio(file_path)
    
    def process_audio(self, file_path):
        try:
            self.show_loading("Predicting speaker count...")
            
            num_speakers = predict_speaker_count(file_path)
            
            self.show_loading(f"Running diarization for {num_speakers} speakers...")
            
            run_diarization(num_speakers, file_path)
            
            self.analyze_results(file_path, num_speakers)
            
            self.hide_loading()
            
        except Exception as e:
            self.hide_loading()
            messagebox.showerror("Error", f"An error occurred: {str(e)}")
    
    def analyze_results(self, file_path, num_speakers):
        # Example dummy stats; replace with real analysis
        self.speaker_stats = [
            {"id": f"Speaker {i+1}", "percentage": np.random.randint(20, 60)} 
            for i in range(num_speakers)
        ]
        
        # Normalize percentages to sum 100
        total = sum(s['percentage'] for s in self.speaker_stats)
        for s in self.speaker_stats:
            s['percentage'] = int((s['percentage'] / total) * 100)
        
        self.display_stats()
        self.display_visualization()
        self.display_output_files(file_path)
    
    def display_stats(self):
        # Clear previous widgets
        for widget in self.stats_tab.winfo_children():
            widget.destroy()
        
        stats_container = tk.Frame(self.stats_tab, bg="#1A1A2E")
        stats_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Display number of speakers
        tk.Label(
            stats_container,
            text=f"Number of Speakers: {len(self.speaker_stats)}",
            font=self.subtitle_font,
            fg="white",
            bg="#1A1A2E"
        ).pack(anchor=tk.W, pady=(0, 20))
        
        # Create pie chart with dark background
        fig, ax = plt.subplots(figsize=(5, 5), facecolor="#1A1A2E")
        ax.set_facecolor("#1A1A2E")
        
        labels = [s['id'] for s in self.speaker_stats]
        sizes = [s['percentage'] for s in self.speaker_stats]
        colors = plt.cm.viridis(np.linspace(0, 1, len(self.speaker_stats)))
        
        wedges, texts, autotexts = ax.pie(
            sizes, 
            labels=labels, 
            colors=colors,
            autopct='%1.1f%%',
            startangle=90,
            textprops={'color': "white"}
        )
        
        for text in texts + autotexts:
            text.set_color('white')
        
        ax.axis('equal')  # Equal aspect ratio ensures pie is circular.
        
        canvas = FigureCanvasTkAgg(fig, master=stats_container)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.LEFT, padx=20)
        
        bars_frame = tk.Frame(stats_container, bg="#1A1A2E")
        bars_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Display percentage bars with correct color conversion
        for i, speaker in enumerate(self.speaker_stats):
            speaker_frame = tk.Frame(bars_frame, bg="#1A1A2E")
            speaker_frame.pack(fill=tk.X, pady=5)
            
            tk.Label(
                speaker_frame,
                text=speaker['id'],
                font=self.subtitle_font,
                fg="white",
                bg="#1A1A2E",
                width=15
            ).pack(side=tk.LEFT)
            
            progress = ttk.Progressbar(
                speaker_frame,
                orient=tk.HORIZONTAL,
                length=200,
                mode='determinate',
                value=speaker['percentage']
            )
            progress.pack(side=tk.LEFT, padx=10)
            
            # Convert matplotlib color RGBA to hex string for Tkinter
            tk.Label(
                speaker_frame,
                text=f"{speaker['percentage']}%",
                font=self.subtitle_font,
                fg=self.rgba_to_hex(colors[i]),  # <-- Correct color format here
                bg="#1A1A2E"
            ).pack(side=tk.LEFT)
    
    def display_visualization(self):
        # Clear previous widgets
        for widget in self.visualization_tab.winfo_children():
            widget.destroy()
        
        fig, ax = plt.subplots(figsize=(10, 3), facecolor="#1A1A2E")
        ax.set_facecolor("#1A1A2E")
        
        time = np.linspace(0, 10, 1000)
        signal = np.sin(2 * np.pi * 5 * time) * (1 + 0.5 * np.sin(2 * np.pi * 0.2 * time))
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(self.speaker_stats)))
        segments = np.array_split(range(len(time)), len(self.speaker_stats))
        
        for i, seg in enumerate(segments):
            ax.plot(time[seg], signal[seg], color=colors[i], linewidth=1.5)
        
        ax.set_xlabel('Time (s)', color='white')
        ax.set_ylabel('Amplitude', color='white')
        ax.tick_params(colors='white')
        
        for spine in ax.spines.values():
            spine.set_color('white')
        
        canvas = FigureCanvasTkAgg(fig, master=self.visualization_tab)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Legend with converted colors
        legend_frame = tk.Frame(self.visualization_tab, bg="#1A1A2E")
        legend_frame.pack(fill=tk.X, padx=20, pady=(0, 20))
        
        for i, speaker in enumerate(self.speaker_stats):
            tk.Label(
                legend_frame,
                text=speaker['id'],
                font=self.subtitle_font,
                fg=self.rgba_to_hex(colors[i]),  # <-- Correct color format here
                bg="#1A1A2E"
            ).pack(side=tk.LEFT, padx=10)
    
    def display_output_files(self, original_path):
        for widget in self.output_tab.winfo_children():
            widget.destroy()
        
        output_dir = os.path.join(os.path.dirname(original_path), 'conca')
        
        if not os.path.exists(output_dir):
            tk.Label(
                self.output_tab,
                text="No output files found",
                font=self.subtitle_font,
                fg="white",
                bg="#1A1A2E"
            ).pack(pady=50)
            return
        
        files_frame = tk.Frame(self.output_tab, bg="#1A1A2E")
        files_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        tk.Label(
            files_frame,
            text="Separated Speaker Audio Files:",
            font=self.subtitle_font,
            fg="white",
            bg="#1A1A2E"
        ).pack(anchor=tk.W, pady=(0, 10))
        
        for i, speaker in enumerate(self.speaker_stats):
            file_frame = tk.Frame(files_frame, bg="#16213E")
            file_frame.pack(fill=tk.X, pady=5)
            
            tk.Label(
                file_frame,
                text=f"{speaker['id']} ({speaker['percentage']}%)",
                font=self.subtitle_font,
                fg="white",
                bg="#16213E",
                width=25
            ).pack(side=tk.LEFT, padx=10)
            
            tk.Button(
                file_frame,
                text="Play",
                font=self.button_font,
                bg="#00F5FF",
                fg="#1A1A2E",
                bd=0,
                command=lambda spk=i: self.play_speaker_audio(spk)
            ).pack(side=tk.LEFT, padx=5)
            
            tk.Button(
                file_frame,
                text="Download",
                font=self.button_font,
                bg="#9457EB",
                fg="white",
                bd=0,
                command=lambda spk=i: self.download_speaker_audio(spk)
            ).pack(side=tk.LEFT, padx=5)
        
        tk.Button(
            files_frame,
            text="Download All Speakers",
            font=self.button_font,
            bg="#00F5FF",
            fg="#1A1A2E",
            bd=0,
            pady=10,
            command=self.download_all_speakers
        ).pack(pady=(20, 0))
    
    def play_audio(self):
        self.playing = True
        self.play_btn.config(text="❚❚", bg="#FF5555")
        messagebox.showinfo("Info", "Audio playback started (simulated)")
    
    def stop_audio(self):
        self.playing = False
        self.play_btn.config(text="▶", bg="#00F5FF")
        messagebox.showinfo("Info", "Audio playback stopped (simulated)")
    
    def play_speaker_audio(self, speaker_idx):
        messagebox.showinfo("Info", f"Playing {self.speaker_stats[speaker_idx]['id']} audio (simulated)")
    
    def download_speaker_audio(self, speaker_idx):
        messagebox.showinfo("Info", f"Downloading {self.speaker_stats[speaker_idx]['id']} audio (simulated)")
    
    def download_all_speakers(self):
        messagebox.showinfo("Info", "Downloading all speaker audio files (simulated)")
    
    def show_loading(self, message):
        self.loading_window = tk.Toplevel(self.root)
        self.loading_window.title("Processing")
        self.loading_window.geometry("300x150")
        self.loading_window.configure(bg="#1A1A2E")
        self.loading_window.resizable(False, False)
        self.loading_window.grab_set()
        
        tk.Label(
            self.loading_window,
            text=message,
            font=self.subtitle_font,
            fg="white",
            bg="#1A1A2E",
            pady=20
        ).pack()
        
        ttk.Progressbar(
            self.loading_window,
            orient=tk.HORIZONTAL,
            mode='indeterminate'
        ).pack(pady=10)
        
        self.loading_window.update()
    
    def hide_loading(self):
        if hasattr(self, 'loading_window'):
            self.loading_window.destroy()
    
    def show_about(self):
        about_text = """Blind Speaker Diarization with Deep Learning

This application uses advanced deep learning techniques to:
1. Identify the number of speakers in an audio file
2. Separate speakers in the audio
3. Provide statistics about each speaker's participation

Developed as a graduation project"""
        
        messagebox.showinfo("About", about_text)

if __name__ == "__main__":
    root = tk.Tk()
    app = ModernInterface(root)
    root.mainloop()
