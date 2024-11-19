"""
Whisper Transcriber
A desktop application for real-time audio transcription using OpenAI's Whisper model.
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import sounddevice as sd
import numpy as np
import faster_whisper
import threading
import queue
import pyperclip
import tempfile
import soundfile as sf
from pathlib import Path
import pystray
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.animation as animation
import os
import sys
import keyboard
import json
from moviepy.editor import VideoFileClip, AudioFileClip
import logging
import psutil
from datetime import datetime

# Configure logging
def setup_logging():
    """Configure application logging"""
    log_dir = Path.home() / '.whisper_transcriber' / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = log_dir / f'whisper_transcriber_{datetime.now().strftime("%Y%m%d")}.log'
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )

def check_system_requirements():
    """Check if system meets minimum requirements"""
    try:
        # Check available memory
        memory = psutil.virtual_memory()
        available_gb = memory.available / (1024 * 1024 * 1024)
        
        # Check CPU cores
        cpu_count = psutil.cpu_count(logical=False)
        
        requirements_met = True
        warnings = []
        
        if available_gb < 4:
            warnings.append(f"Low memory available: {available_gb:.1f}GB. Recommended: 4GB+")
            requirements_met = False
            
        if cpu_count < 2:
            warnings.append(f"Limited CPU cores: {cpu_count}. Recommended: 2+")
            requirements_met = False
            
        return requirements_met, warnings
        
    except Exception as e:
        logging.error(f"Error checking system requirements: {e}")
        return True, []

def cleanup_temp_files():
    """Clean up old temporary files"""
    try:
        temp_dir = Path.home() / '.whisper_transcriber' / 'temp'
        if temp_dir.exists():
            current_time = datetime.now().timestamp()
            for file in temp_dir.glob('*'):
                # Remove files older than 24 hours
                if current_time - file.stat().st_mtime > 86400:
                    file.unlink()
    except Exception as e:
        logging.error(f"Error cleaning up temp files: {e}")

class Theme:
    def __init__(self):
        # Define a cohesive light gray theme with good contrast ratios
        self.theme = {
            # Main application colors
            "bg": "#f0f0f0",               # Light gray background
            "fg": "#000000",               # Black text for readability
            "accent": "#007acc",           # Bright blue accent

            # Plot/visualization colors
            "plot_bg": "#e6e6e6",         # Slightly darker than main bg
            "plot_line": "#005500",       # Dark green for visibility
            "plot_grid": "#cccccc",       # Light gray grid

            # System tray icons
            "tray_icons": {
                "idle": "#007acc",         # Blue
                "recording": "#cc0000",    # Dark red
                "processing": "#e68a00"    # Amber
            },

            # UI component colors
            "popup_bg": "#ffffff",         # White background for popups
            "popup_fg": "#000000",         # Black text
            "frame_bg": "#e6e6e6",         # Slightly darker than main bg
            "button_bg": "#d9d9d9",        # Light gray buttons
            "button_fg": "#000000",        # Black text
            "entry_bg": "#ffffff",         # White input fields
            "entry_fg": "#000000",         # Black text

            # Additional UI elements
            "border": "#a6a6a6",           # Medium gray border
            "selection_bg": "#007acc",     # Bright blue for selected item background
            "hover_bg": "#d0d0d0",         # Slightly darker gray for hover state
            "disabled_bg": "#e0e0e0",      # Disabled element background
            "disabled_fg": "#a0a0a0",      # Disabled element text

            # Text field specific
            "text_selected_bg": "#007acc", # Selected text background
            "text_selected_fg": "#ffffff", # Selected text color
            "text_inactive_bg": "#d9d9d9", # Inactive text field background
            "text_inactive_fg": "#7a7a7a"  # Inactive text color
        }


    def get_color(self, key):
        return self.theme.get(key)

    def apply_ttk_styles(self, style):
        """Apply theme to ttk widgets"""
        # Configure common styles
        style.configure(".",
            background=self.get_color("bg"),
            foreground=self.get_color("fg"),
            fieldbackground=self.get_color("entry_bg"),
            selectbackground=self.get_color("selection_bg"),
            selectforeground=self.get_color("fg")
        )
        
        # Frame styling
        style.configure("TFrame",
            background=self.get_color("frame_bg")
        )
        
        # Label styling
        style.configure("TLabel",
            background=self.get_color("frame_bg"),
            foreground=self.get_color("fg")
        )
        
        # Button styling
        style.configure("TButton",
            background=self.get_color("button_bg"),
            foreground=self.get_color("button_fg"),
            bordercolor=self.get_color("border")
        )
        style.map("TButton",
            background=[("active", self.get_color("hover_bg"))],
            foreground=[("disabled", self.get_color("disabled_fg"))]
        )
        
        # Entry styling
        style.configure("TEntry",
            fieldbackground=self.get_color("entry_bg"),
            foreground=self.get_color("entry_fg"),
            selectbackground=self.get_color("text_selected_bg"),
            selectforeground=self.get_color("text_selected_fg")
        )
        
        # Combobox styling
        style.configure("TCombobox",
            fieldbackground=self.get_color("entry_bg"),
            background=self.get_color("button_bg"),
            foreground=self.get_color("fg"),
            arrowcolor=self.get_color("fg")
        )
        
        # Checkbutton styling
        style.configure("TCheckbutton",
            background=self.get_color("frame_bg"),
            foreground=self.get_color("fg")
        )
        
        # Progressbar styling
        style.configure("TProgressbar",
            background=self.get_color("accent"),
            troughcolor=self.get_color("entry_bg")
        )
        
        # Scrollbar styling
        style.configure("TScrollbar",
            background=self.get_color("button_bg"),
            troughcolor=self.get_color("frame_bg"),
            arrowcolor=self.get_color("fg")
        )
        
        # LabelFrame styling
        style.configure("TLabelframe",
            background=self.get_color("frame_bg"),
            foreground=self.get_color("fg")
        )
        style.configure("TLabelframe.Label",
            background=self.get_color("frame_bg"),
            foreground=self.get_color("fg")
        )

    def apply_text_widget_colors(self, text_widget):
        """Apply theme to Text widget"""
        text_widget.configure(
            bg=self.get_color("entry_bg"),
            fg=self.get_color("fg"),
            insertbackground=self.get_color("fg"),
            selectbackground=self.get_color("text_selected_bg"),
            selectforeground=self.get_color("text_selected_fg"),
            inactiveselectbackground=self.get_color("text_inactive_bg")
        )

    def apply_plot_colors(self, fig, ax):
        """Apply theme to matplotlib plot"""
        fig.set_facecolor(self.get_color("plot_bg"))
        ax.set_facecolor(self.get_color("plot_bg"))
        ax.tick_params(colors=self.get_color("fg"))
        ax.xaxis.label.set_color(self.get_color("fg"))
        ax.yaxis.label.set_color(self.get_color("fg"))
        if ax.title:
            ax.title.set_color(self.get_color("fg"))
        ax.grid(True, color=self.get_color("plot_grid"), linestyle='--', alpha=0.5)

class WhisperTranscriberApp:
    def __init__(self):
        # Initialize the main window
        self.root = tk.Tk()
        self.root.withdraw()

        # Initialize theme
        self.theme = Theme()
        style = ttk.Style()
        self.theme.apply_ttk_styles(style)

        # Define supported file types
        self.supported_audio = (".mp3", ".wav", ".m4a", ".flac")
        self.supported_video = (".mp4", ".avi", ".mkv", ".mov")
        
        # Initialize models dictionary
        self.models = {
            "Tiny (fastest, least accurate)": "tiny",
            "Tiny.en (English optimized)": "tiny.en",
            "Base (fast, balanced)": "base",
            "Base.en (English optimized)": "base.en",
            "Small (balanced)": "small",
            "Small.en (English optimized)": "small.en",
            "Medium (accurate)": "medium",
            "Medium.en (English optimized)": "medium.en",
            "Large-v1 (most accurate, slowest)": "large-v1",
            "Large-v2 (most accurate, newest)": "large-v2",
            "Large-v3 (latest, most accurate)": "large-v3"
        }

        self.key_mapping = {
            'control_l': 'left ctrl',
            'control_r': 'right ctrl',
            'shift_l': 'left shift',
            'shift_r': 'right shift',
            'alt_l': 'left alt',
            'alt_r': 'right alt',
            'escape': 'esc'
        }

        # Initialize queues for thread-safe communication
        self.command_queue = queue.Queue()
        self.status_queue = queue.Queue()
        self.audio_queue = queue.Queue()
        
        # Initialize configuration
        self.default_config = {
            'toggle_hotkey': 'ctrl+shift+r',
            'start_minimized': True,
        }
        self.config_file = Path.home() / '.whisper_transcriber_config.json'
        self.config = self.default_config.copy()
        self.load_config()

        # Initialize variables
        self.setup_variables()
        
        # Create UI and start the application
        self.create_tray_icon()
        self.create_window()
        
        # Set up periodic checks for queues
        self.root.after(100, self.process_command_queue)
        self.root.after(100, self.process_status_queue)
        
        # Load initial model in background
        self.root.after(0, self.load_initial_model)
        
        # Set up hotkeys
        self.setup_hotkeys()
        
    def create_hotkeys_frame(self, parent):
        """Create the hotkeys configuration section"""
        hotkeys_frame = ttk.LabelFrame(parent, text="Hotkeys", padding="5")
        hotkeys_frame.grid(row=8, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)
        hotkeys_frame.grid_columnconfigure(1, weight=1)

        ttk.Label(hotkeys_frame, text="Toggle Recording:").grid(row=0, column=0, sticky=tk.W, pady=2)
        
        self.hotkey_entry = ttk.Entry(hotkeys_frame, textvariable=self.toggle_hotkey_var, state='readonly')
        self.hotkey_entry.grid(row=0, column=1, sticky=tk.W, pady=2)
        
        button_frame = ttk.Frame(hotkeys_frame)
        button_frame.grid(row=0, column=2, padx=5)
        
        ttk.Button(button_frame, text="Set", command=self.set_hotkey).grid(row=0, column=0)
        ttk.Button(button_frame, text="Clear", command=self.clear_hotkey).grid(row=0, column=1)
        
        ttk.Button(hotkeys_frame, text="Reset to Default", 
                  command=self.reset_hotkeys).grid(row=1, column=0, columnspan=3, pady=5)
        
        self.hotkey_entry.bind('<KeyPress>', self.on_hotkey_press)
        self.hotkey_entry.bind('<KeyRelease>', self.on_hotkey_release)
    
    def create_status_label(self, parent):
        """Create the status label"""
        self.status_label = ttk.Label(parent, textvariable=self.status_var)
        self.status_label.grid(row=6, column=0, columnspan=3, pady=5)

    def on_model_change(self, event):
        """Handle model selection changes"""
        model_name = self.models[self.model_var.get()]
        if model_name != self.current_model:
            def load_model():
                try:
                    self.update_status(f"Loading {model_name} model...")
                    self.model = faster_whisper.WhisperModel(model_name, device="cpu", compute_type="int8")
                    self.current_model = model_name
                    self.update_status("Model loaded successfully")
                except Exception as e:
                    self.update_status(f"Error loading model: {str(e)}")

            threading.Thread(target=load_model, daemon=True).start()

    def load_initial_model(self):
        """Load the initial whisper model"""
        def load_model():
            try:
                model_name = self.models[self.model_var.get()]
                self.update_status(f"Loading {model_name} model...")
                self.model = faster_whisper.WhisperModel(model_name, device="cpu", compute_type="int8")
                self.current_model = model_name
                self.update_status("Model loaded successfully")
            except Exception as e:
                self.update_status(f"Error loading model: {str(e)}")

        threading.Thread(target=load_model, daemon=True).start()

    def toggle_recording(self):
        """Toggle recording state"""
        if self.recording_var.get():
            self.start_recording()
        else:
            self.stop_recording()

    def toggle_recording_hotkey(self):
        """Handle hotkey press"""
        self.root.after(0, lambda: self.recording_var.set(not self.recording_var.get()))
        self.root.after(0, self.toggle_recording)

    def setup_hotkeys(self):
        """Set up keyboard hotkeys"""
        try:
            keyboard.unhook_all()
            mapped_hotkey = '+'.join(self.key_mapping.get(key, key) for key in self.config['toggle_hotkey'].split('+'))
            keyboard.add_hotkey(mapped_hotkey, self.toggle_recording_hotkey)
        except Exception as e:
            self.update_status(f"Error setting up hotkeys: {str(e)}")

    def set_hotkey(self, hotkey_type):
        """Open dialog to set new hotkey"""
        current = self.config[hotkey_type]
        HotkeyDialog(self.root, current, 
                    lambda new_hotkey: self.save_hotkey(new_hotkey),
                    self.theme, self.key_mapping)

    def save_hotkey(self, new_hotkey):
        """Save the current hotkey and close the dialog"""
        self.config['toggle_hotkey'] = new_hotkey
        self.toggle_hotkey_var.set(new_hotkey)
        self.save_config()
        self.setup_hotkeys()

    def reset_hotkeys(self):
        """Reset hotkeys to defaults"""
        if messagebox.askyesno("Reset Hotkey", "Reset hotkey to default value?"):
            self.config['toggle_hotkey'] = self.default_config['toggle_hotkey']
            self.toggle_hotkey_var.set(self.config['toggle_hotkey'])
            self.save_config()
            self.setup_hotkeys()

    def copy_to_clipboard(self):
        """Copy transcription to clipboard"""
        text = self.text_output.get(1.0, tk.END).strip()
        if text:
            pyperclip.copy(text)
            self.update_status("Copied to clipboard")
        else:
            self.update_status("No text to copy")

    def load_config(self):
        """Load configuration from file"""
        try:
            if self.config_file.exists():
                with open(self.config_file, 'r') as f:
                    stored_config = json.load(f)
                    self.config.update(stored_config)
            self.save_config()  # Save to ensure all default values are written
        except Exception as e:
            logging.error(f"Error loading config: {e}")

    def save_config(self):
        """Save configuration to file"""
        try:
            # Create parent directory if it doesn't exist
            self.config_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=4)
        except Exception as e:
            logging.error(f"Error saving config: {e}")

    def toggle_window(self):
        """Toggle window visibility"""
        def toggle():
            if self.root.state() == 'withdrawn':
                self.show_window()
            else:
                self.hide_window()
        self.root.after(0, toggle)

    def show_window(self):
        """Show and restore the window"""
        self.root.deiconify()
        self.root.lift()
        self.root.focus_force()

    def hide_window(self):
        """Hide the window"""
        self.root.withdraw()

    def toggle_start_minimized(self, item):
        """Toggle start minimized setting"""
        self.config['start_minimized'] = not self.config['start_minimized']
        item.checked = self.config['start_minimized']  # Update the checked state
        self.save_config()

    def import_audio(self):
        """Handle importing audio files"""
        filetypes = [
            ("Audio Files", " ".join(f"*{ext}" for ext in self.supported_audio)),
            ("All Files", "*.*")
        ]
        
        filename = filedialog.askopenfilename(
            title="Select Audio File",
            filetypes=filetypes
        )
        
        if filename:
            threading.Thread(target=lambda: self.process_audio_file(filename), 
                           daemon=True).start()

    def import_video(self):
        """Handle importing video files"""
        filetypes = [
            ("Video Files", " ".join(f"*{ext}" for ext in self.supported_video)),
            ("All Files", "*.*")
        ]
        
        filename = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=filetypes
        )
        
        if filename:
            threading.Thread(target=lambda: self.convert_video_to_audio(filename), 
                           daemon=True).start()

    def process_audio_file(self, audio_path, temp_file=False):
        """Process an audio file for transcription"""
        try:
            self.update_status("Processing audio file...")
            self.root.after(0, lambda: self.progress_var.set(0))
            
            def process_audio():
                try:
                    # Load the audio file
                    audio = AudioFileClip(audio_path)
                    
                    # Transcribe using the model
                    segments, _ = self.model.transcribe(audio_path)
                    text = " ".join([segment.text for segment in segments]).strip()
                    
                    # Clean up
                    audio.close()
                    if temp_file:
                        Path(audio_path).unlink()
                    
                    self.queue_command('update_transcription', text)
                    
                except Exception as e:
                    self.queue_command('handle_processing_error', f"Error processing audio: {str(e)}")
            
            threading.Thread(target=process_audio, daemon=True).start()
            
        except Exception as e:
            self.update_status(f"Error loading audio file: {str(e)}")
            self.root.after(0, lambda: self.progress_var.set(0))

    def convert_video_to_audio(self, video_path):
        """Convert video to audio and process it"""
        try:
            self.update_status("Converting video to audio...")
            self.root.after(0, lambda: self.progress_var.set(0))
            
            # Create temp directory
            temp_dir = Path.home() / '.whisper_transcriber' / 'temp'
            temp_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate output path
            audio_path = temp_dir / f"temp_audio_{os.path.splitext(os.path.basename(video_path))[0]}.mp3"
            
            def conversion_progress(t):
                """Update progress bar in main thread"""
                self.root.after(0, lambda: self.progress_var.set(min(100, t * 100)))
            
            def convert():
                try:
                    # Convert video to audio
                    video = VideoFileClip(video_path)
                    audio = video.audio
                    audio.write_audiofile(str(audio_path), 
                                        fps=self.sample_rate,
                                        nbytes=2,
                                        codec='libmp3lame',
                                        progress_bar=False,
                                        progress_callback=conversion_progress)
                    
                    # Clean up
                    video.close()
                    audio.close()
                    
                    # Process the audio file
                    self.queue_command('process_audio_file', str(audio_path), True)
                    
                except Exception as e:
                    self.queue_command('handle_processing_error', f"Error converting video: {str(e)}")
                    self.root.after(0, lambda: self.progress_var.set(0))
            
            threading.Thread(target=convert, daemon=True).start()
            
        except Exception as e:
            self.update_status(f"Error preparing video conversion: {str(e)}")
            self.root.after(0, lambda: self.progress_var.set(0))

    def setup_variables(self):
        """Initialize all application variables"""
        self.recording = False
        self.audio_data = []
        self.sample_rate = 16000
        self.buffer_size = 4096
        self.stream = None
        self.recording_thread = None
        self.model = None
        self.current_model = None
        self.animation = None
        
        # UI Variables (must be created after root initialization)
        self.model_var = tk.StringVar(self.root, value="Base (fast, balanced)")
        self.device_var = tk.StringVar(self.root)
        self.auto_copy_var = tk.BooleanVar(self.root, value=True)
        self.recording_var = tk.BooleanVar(self.root, value=False)
        self.status_var = tk.StringVar(self.root, value="Ready")
        self.progress_var = tk.DoubleVar(self.root)
        self.toggle_hotkey_var = tk.StringVar(self.root, value=self.config['toggle_hotkey'])
        
        # Initialize audio visualization data
        self.audio_data_display = np.zeros(self.buffer_size)
        self.plot_data = np.zeros(self.buffer_size)
        
        # Get available audio devices
        self.audio_devices = self.get_audio_devices()
        if self.audio_devices:
            self.device_var.set(list(self.audio_devices.keys())[0])

    def process_command_queue(self):
        """Process commands from other threads"""
        try:
            while True:
                command, args = self.command_queue.get_nowait()
                if hasattr(self, command):
                    getattr(self, command)(*args)
        except queue.Empty:
            pass
        finally:
            self.root.after(100, self.process_command_queue)

    def process_status_queue(self):
        """Process status updates from other threads"""
        try:
            while True:
                status = self.status_queue.get_nowait()
                self.status_var.set(status)
        except queue.Empty:
            pass
        finally:
            self.root.after(100, self.process_status_queue)

    def queue_command(self, command, *args):
        """Queue a command to be executed in the main thread"""
        self.command_queue.put((command, args))

    def update_status(self, status):
        """Queue a status update to be shown in the main thread"""
        self.status_queue.put(status)

    def start_recording(self):
        """Start recording from the main thread"""
        if not self.model:
            self.update_status("Please wait for model to load")
            self.recording_var.set(False)
            return
            
        if self.recording:
            return

        self.recording = True
        self.audio_data = []
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break

        # Update UI
        self.update_status("Recording...")
        self.create_tray_icon_image('recording')
        self.tray_icon.icon = self.icon_image
        self.reset_visualization()

        # Start recording thread
        self.recording_thread = threading.Thread(target=self.record_audio, daemon=True)
        self.recording_thread.start()

    def record_audio(self):
        """Record audio in a separate thread"""
        try:
            device_name = self.device_var.get()
            device_idx = self.audio_devices[device_name]
            
            def audio_callback(indata, frames, time, status):
                if status:
                    print(status)
                if self.recording:
                    self.audio_queue.put(indata.copy())
                    self.audio_data.append(indata.copy())

            with sd.InputStream(
                callback=audio_callback,
                channels=1,
                samplerate=self.sample_rate,
                device=device_idx,
                blocksize=1024
            ) as self.stream:
                while self.recording:
                    sd.sleep(50)
                    
        except Exception as e:
            self.queue_command('handle_recording_error', str(e))

    def handle_recording_error(self, error_message):
        """Handle recording errors in the main thread"""
        self.recording = False
        self.recording_var.set(False)
        self.update_status(f"Recording error: {error_message}")
        self.create_tray_icon_image('idle')
        self.tray_icon.icon = self.icon_image

    def stop_recording(self):
        """Stop recording from the main thread"""
        if not self.recording:
            return

        self.recording = False
        self.recording_var.set(False)
        self.update_status("Processing audio...")
        
        # Update UI
        self.create_tray_icon_image('processing')
        self.tray_icon.icon = self.icon_image
        
        if self.animation:
            self.animation.event_source.stop()
            self.animation = None
        
        if self.stream:
            self.stream.close()
            self.stream = None
        
        # Start processing in a new thread
        threading.Thread(target=self.process_audio, daemon=True).start()

    def process_audio(self):
        """Process recorded audio in a separate thread"""
        try:
            if not self.audio_data or len(self.audio_data) == 0:
                self.queue_command('handle_processing_error', "No audio data recorded")
                return

            try:
                audio = np.concatenate(self.audio_data)
            except ValueError:
                self.queue_command('handle_processing_error', "Error: No audio data to process")
                return
            
            if len(audio) < 100:
                self.queue_command('handle_processing_error', "Recording too short")
                return

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                sf.write(temp_file.name, audio, self.sample_rate)
                segments, _ = self.model.transcribe(temp_file.name)
                text = " ".join([segment.text for segment in segments]).strip()
                self.queue_command('update_transcription', text)
            
            Path(temp_file.name).unlink()
            
        except Exception as e:
            self.queue_command('handle_processing_error', f"Error during transcription: {str(e)}")

    def update_transcription(self, text):
        """Update transcription text in the main thread"""
        self.text_output.delete(1.0, tk.END)
        self.text_output.insert(tk.END, text)
        
        if self.auto_copy_var.get():
            pyperclip.copy(text)
            self.update_status("Transcription complete and copied to clipboard")
        else:
            self.update_status("Transcription complete")
        
        self.create_tray_icon_image('idle')
        self.tray_icon.icon = self.icon_image

    def handle_processing_error(self, error_message):
        """Handle processing errors in the main thread"""
        self.update_status(error_message)
        self.create_tray_icon_image('idle')
        self.tray_icon.icon = self.icon_image

    def create_window(self):
        """Create the main application window"""
        # Configure root window
        self.root.title("Whisper Transcriber")
        self.root.protocol('WM_DELETE_WINDOW', self.hide_window)
        
        # Configure grid weight for main window
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_rowconfigure(0, weight=1)

        # Main frame configuration
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        main_frame.grid_columnconfigure(0, weight=1)
        main_frame.grid_columnconfigure(1, weight=1)

        # Import frame
        self.create_import_frame(main_frame)
        
        # Model selection frame
        self.create_model_frame(main_frame)
        
        # Audio device frame
        self.create_device_frame(main_frame)
        
        # Auto-copy checkbox
        self.create_auto_copy_checkbox(main_frame)
        
        # Audio visualization
        self.setup_audio_visualization(main_frame)
        
        # Record toggle button
        self.create_record_button(main_frame)
        
        # Status label
        self.create_status_label(main_frame)
        
        # Transcription frame
        self.create_transcription_frame(main_frame)
        
        # Hotkeys frame
        self.create_hotkeys_frame(main_frame)

    def create_import_frame(self, parent):
        """Create the import section of the UI"""
        import_frame = ttk.LabelFrame(parent, text="Import", padding="5")
        import_frame.grid(row=0, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)
        import_frame.grid_columnconfigure(0, weight=1)
        import_frame.grid_columnconfigure(1, weight=1)
        
        ttk.Button(import_frame, text="Import Audio File", 
                  command=self.import_audio).grid(row=0, column=0, padx=5, pady=5, sticky=(tk.W, tk.E))
        ttk.Button(import_frame, text="Import Video File", 
                  command=self.import_video).grid(row=0, column=1, padx=5, pady=5, sticky=(tk.W, tk.E))
        
        self.progress_bar = ttk.Progressbar(import_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), padx=5, pady=5)

    def create_model_frame(self, parent):
        """Create the model selection section"""
        model_frame = ttk.Frame(parent)
        model_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)
        model_frame.grid_columnconfigure(1, weight=1)
        
        ttk.Label(model_frame, text="Model:").grid(row=0, column=0, padx=5)
        
        model_dropdown = ttk.Combobox(model_frame, textvariable=self.model_var, 
                                    values=list(self.models.keys()))
        model_dropdown.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=5)
        model_dropdown.set("Base (fast, balanced)")
        model_dropdown.bind('<<ComboboxSelected>>', self.on_model_change)
        
        ttk.Button(model_frame, text="?", width=3,
                  command=self.show_model_info).grid(row=0, column=2, padx=(5, 0))

    def create_device_frame(self, parent):
        """Create the audio device selection section"""
        device_frame = ttk.Frame(parent)
        device_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)
        device_frame.grid_columnconfigure(1, weight=1)
        
        ttk.Label(device_frame, text="Input Device:").grid(row=0, column=0, padx=5)
        
        device_dropdown = ttk.Combobox(device_frame, textvariable=self.device_var, 
                                     values=list(self.audio_devices.keys()))
        device_dropdown.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=5)
        if self.audio_devices:
            device_dropdown.set(list(self.audio_devices.keys())[0])

    def create_auto_copy_checkbox(self, parent):
        """Create the auto-copy checkbox"""
        ttk.Checkbutton(parent, text="Auto-copy to clipboard", 
                       variable=self.auto_copy_var).grid(row=3, column=0, 
                       columnspan=3, sticky=tk.W, pady=5)

    def setup_audio_visualization(self, parent):
        """Set up the audio visualization display"""
        viz_frame = ttk.Frame(parent)
        viz_frame.grid(row=4, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)
        viz_frame.grid_columnconfigure(0, weight=1)
        
        self.fig, self.ax = plt.subplots(figsize=(6, 2))
        self.theme.apply_plot_colors(self.fig, self.ax)
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=viz_frame)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky=(tk.W, tk.E))
        
        x = np.linspace(0, self.buffer_size - 1, self.buffer_size)
        self.line, = self.ax.plot(x, self.plot_data, linewidth=1)
        
        self.ax.set_ylim(-1, 1)
        self.ax.set_xlim(0, self.buffer_size - 1)
        self.ax.set_title('Audio Input', pad=2)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        
        self.fig.tight_layout()
        self.animation = None

    def update_audio_visualization(self, frame):
        """Update the audio visualization (called by animation)"""
        if not self.recording:
            return self.line,

        try:
            data = []
            while not self.audio_queue.empty():
                chunk = self.audio_queue.get_nowait()
                data.append(chunk)
            
            if data:
                new_data = np.concatenate(data).flatten()
                max_val = np.abs(new_data).max()
                if max_val > 0:
                    new_data = new_data / max_val
                
                data_len = len(new_data)
                roll_amount = min(data_len, self.buffer_size)
                
                self.plot_data = np.roll(self.plot_data, -roll_amount)
                
                if data_len >= self.buffer_size:
                    self.plot_data[-self.buffer_size:] = new_data[-self.buffer_size:]
                else:
                    self.plot_data[-data_len:] = new_data
                
                self.line.set_ydata(self.plot_data)
                
                # Only draw if recording
                if self.recording:
                    self.fig.canvas.draw_idle()
                    self.fig.canvas.flush_events()
        except Exception as e:
            self.queue_command('handle_visualization_error', str(e))
        
        return self.line,

    def reset_visualization(self):
        """Reset the visualization in the main thread"""
        self.plot_data = np.zeros(self.buffer_size)
        self.line.set_ydata(self.plot_data)
        
        if self.animation:
            self.animation.event_source.stop()
        
        self.animation = animation.FuncAnimation(
            self.fig,
            self.update_audio_visualization,
            interval=30,
            blit=True,
            cache_frame_data=False
        )

    def create_record_button(self, parent):
        """Create the record toggle button"""
        self.record_toggle = ttk.Checkbutton(
            parent,
            text="Record",
            variable=self.recording_var,
            style='Toggle.TCheckbutton',
            command=self.toggle_recording
        )
        self.record_toggle.grid(row=5, column=0, columnspan=3, pady=10)

        def create_status_label(self, parent):
            """Create the status label"""
            status_label = ttk.Label(parent, textvariable=self.status_var)
            status_label.grid(row=6, column=0, columnspan=3, pady=5)

    def create_transcription_frame(self, parent):
        """Create the transcription output section"""
        transcription_frame = ttk.Frame(parent)
        transcription_frame.grid(row=7, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S))
        transcription_frame.grid_columnconfigure(0, weight=1)
        transcription_frame.grid_rowconfigure(1, weight=1)

        ttk.Label(transcription_frame, text="Transcription:").grid(row=0, column=0, sticky=tk.W, pady=(5,0))
        
        text_frame = ttk.Frame(transcription_frame)
        text_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        text_frame.grid_columnconfigure(0, weight=1)
        text_frame.grid_rowconfigure(0, weight=1)

        self.text_output = tk.Text(text_frame, height=10, width=50, wrap=tk.WORD)
        self.theme.apply_text_widget_colors(self.text_output)
        self.text_output.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        scrollbar = ttk.Scrollbar(text_frame, orient="vertical", command=self.text_output.yview)
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.text_output.configure(yscrollcommand=scrollbar.set)

        self.copy_button = ttk.Button(transcription_frame, text="Copy to Clipboard", 
                                    command=self.copy_to_clipboard)
        self.copy_button.grid(row=2, column=0, pady=5)

    def create_hotkeys_frame(self, parent):
        """Create the hotkeys configuration section"""
        hotkeys_frame = ttk.LabelFrame(parent, text="Hotkeys", padding="5")
        hotkeys_frame.grid(row=8, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)
        hotkeys_frame.grid_columnconfigure(1, weight=1)

        ttk.Label(hotkeys_frame, text="Toggle Recording:").grid(row=0, column=0, sticky=tk.W, pady=2)
        ttk.Label(hotkeys_frame, textvariable=self.toggle_hotkey_var).grid(row=0, column=1, sticky=tk.W, pady=2)
        ttk.Button(hotkeys_frame, text="Set", 
                  command=lambda: self.set_hotkey('toggle_hotkey')).grid(row=0, column=2, padx=5)
        
        ttk.Button(hotkeys_frame, text="Reset to Default", 
                  command=self.reset_hotkeys).grid(row=1, column=0, columnspan=3, pady=5)

    def show_model_info(self):
        """Show model information dialog"""
        ModelInfoDialog(self.root, self.theme)

    def create_tray_icon(self):
        """Create the system tray icon"""
        self.create_tray_icon_image('idle')
        
        menu = (
            pystray.MenuItem("Show/Hide", self.toggle_window),
            pystray.MenuItem("Start Minimized", 
                             lambda item: self.toggle_start_minimized(item), 
                             checked=lambda item: self.config['start_minimized']),
            pystray.MenuItem("Exit", self.quit_application)
        )
        
        self.tray_icon = pystray.Icon("whisper_transcriber", self.icon_image, "Whisper Transcriber", menu)

    def create_tray_icon_image(self, state):
        """Create the tray icon image"""
        icon_size = 64
        self.icon_image = Image.new('RGB', (icon_size, icon_size), color='white')
        draw = ImageDraw.Draw(self.icon_image)
        
        colors = {
            'idle': self.theme.get_color('tray_icons')['idle'],
            'recording': self.theme.get_color('tray_icons')['recording'],
            'processing': self.theme.get_color('tray_icons')['processing']
        }
        
        color = colors[state]
        if color.startswith('#'):
            color = tuple(int(color[i:i+2], 16) for i in (1, 3, 5))
        
        draw.ellipse([4, 4, icon_size-4, icon_size-4], fill=color)

    def get_audio_devices(self):
        """Get list of available audio input devices"""
        devices = sd.query_devices()
        input_devices = {}
        for i, device in enumerate(devices):
            if device['max_input_channels'] > 0:
                input_devices[f"{device['name']}"] = i
        return input_devices

    def run(self):
        """Start the application"""
        if not self.config.get('start_minimized', False):
            self.show_window()
        else:
            self.root.withdraw()
        
        # Start system tray icon in separate thread
        tray_thread = threading.Thread(target=self.tray_icon.run, daemon=True)
        tray_thread.start()
        
        # Start main loop
        self.root.mainloop()

    def quit_application(self):
        """Clean shutdown of the application"""
        self.recording = False
        
        if self.stream:
            self.stream.close()
        
        try:
            keyboard.unhook_all()
        except:
            pass
        
        if hasattr(self, 'tray_icon'):
            self.tray_icon.stop()
        
        self.root.quit()
        
class HotkeyDialog(tk.Toplevel):
    """Dialog for capturing and setting hotkeys"""
    def __init__(self, parent, current_hotkey, on_save, theme, key_mapping):
        super().__init__(parent)
        self.key_mapping = key_mapping
        
        self.title("Set Hotkey")
        self.resizable(False, False)
        
        self.theme = theme
        self.current_hotkey = current_hotkey
        self.on_save = on_save
        self.pressed_keys = set()
        
        # Make dialog modal
        self.transient(parent)
        self.grab_set()
        
        # Setup UI
        self.setup_ui()
        
        # Center dialog on parent
        self.geometry("+%d+%d" % (
            parent.winfo_rootx() + 50,
            parent.winfo_rooty() + 50))

    def setup_ui(self):
        """Setup the dialog UI"""
        style = ttk.Style()
        colors = self.theme.theme
        
        # Configure styles
        style.configure('Dialog.TFrame', 
                       background=colors['popup_bg'])
        style.configure('Dialog.TLabel', 
                       background=colors['popup_bg'],
                       foreground=colors['popup_fg'])
        style.configure('Dialog.TButton',
                       background=colors['button_bg'],
                       foreground=colors['button_fg'])
        
        # Main frame
        frame = ttk.Frame(self, padding="10", style='Dialog.TFrame')
        frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Instructions
        ttk.Label(frame, 
                 text="Press the desired key combination:",
                 style='Dialog.TLabel').grid(row=0, column=0, columnspan=2, pady=5)
        
        # Hotkey display
        self.hotkey_var = tk.StringVar(value=self.current_hotkey)
        self.hotkey_entry = ttk.Entry(frame, 
                                    textvariable=self.hotkey_var,
                                    state='readonly',
                                    width=30)
        self.hotkey_entry.grid(row=1, column=0, columnspan=2, pady=5, padx=5)
        
        # Buttons
        button_frame = ttk.Frame(frame, style='Dialog.TFrame')
        button_frame.grid(row=2, column=0, columnspan=2, pady=10)
        
        ttk.Button(button_frame, 
                  text="Clear", 
                  command=self.clear_hotkey,
                  style='Dialog.TButton').grid(row=0, column=0, padx=5)
        ttk.Button(button_frame, 
                  text="Save", 
                  command=self.save_hotkey,
                  style='Dialog.TButton').grid(row=0, column=1, padx=5)
        
        # Bind key events
        self.bind('<KeyPress>', self.on_key_press)
        self.bind('<KeyRelease>', self.on_key_release)

    def clear_hotkey(self):
        """Clear the current hotkey"""
        self.pressed_keys.clear()
        self.hotkey_var.set('')

    def save_hotkey(self):
        """Save the current hotkey and close the dialog"""
        self.on_save(self.hotkey_var.get())
        self.destroy()

    def on_key_press(self, event):
        """Record the pressed key"""
        key = event.keysym.lower()
        if key not in self.pressed_keys:
            self.pressed_keys.add(key)
            self.update_hotkey_display()

    def on_key_release(self, event):
        """Remove the released key"""
        key = event.keysym.lower()
        if key in self.pressed_keys:
            self.pressed_keys.remove(key)
            self.update_hotkey_display()

    def update_hotkey_display(self):
        """Update the hotkey display"""
        modifier_order = {'ctrl': 0, 'shift': 1, 'alt': 2}
        
        modifiers = sorted(
            [k for k in self.pressed_keys if k in modifier_order],
            key=lambda x: modifier_order[x]
        )
        other_keys = [k for k in self.pressed_keys if k not in modifier_order]
        
        all_keys = modifiers + other_keys
        if all_keys:
            self.hotkey_var.set('+'.join(all_keys))
        else:
            self.hotkey_var.set('')

class ModelInfoDialog(tk.Toplevel):
    """Dialog for displaying model information"""
    def __init__(self, parent, theme):
        super().__init__(parent)
        self.title("Model Information")
        self.theme = theme
        self.setup_dialog()
        self.transient(parent)
        self.grab_set()
        self.geometry(f"+{parent.winfo_rootx() + 50}+{parent.winfo_rooty() + 50}")

    def setup_dialog(self):
        """Setup the dialog UI"""
        colors = self.theme.theme
        self.configure(bg=colors['bg'])
        
        frame = ttk.Frame(self, padding="20")
        frame.pack(fill=tk.BOTH, expand=True)
        
        info_text = """Model Information:
        
        Tiny/Tiny.en: Fastest, lowest memory usage, least accurate
        - Good for: Quick transcriptions where perfect accuracy isn't critical
        - Memory: ~1GB

        Base/Base.en: Good balance of speed and accuracy
        - Good for: General purpose transcription
        - Memory: ~1.5GB

        Small/Small.en: Better accuracy, still reasonably fast
        - Good for: Most everyday transcription needs
        - Memory: ~2GB

        Medium/Medium.en: High accuracy, slower processing
        - Good for: When accuracy is important and time isn't critical
        - Memory: ~5GB

        Large-v1/v2/v3: Highest accuracy, slowest processing
        - Good for: When maximum accuracy is required
        - Memory: ~10GB

        Note: '.en' models are optimized for English and may perform better for English-only content."""
        
        text_widget = tk.Text(frame, wrap=tk.WORD, width=60, height=20,
                            bg=colors['popup_bg'],
                            fg=colors['popup_fg'])
        text_widget.insert('1.0', info_text)
        text_widget.configure(state='disabled')
        text_widget.pack(fill=tk.BOTH, expand=True)
        
        ttk.Button(frame, text="Close", command=self.destroy).pack(pady=10)

def is_admin():
    """Check if the application is running with admin privileges"""
    try:
        # Try posix first for macOS and Linux
        return os.getuid() == 0
    except AttributeError:
        # If posix fails, try windows
        try:
            import ctypes
            return ctypes.windll.shell32.IsUserAnAdmin() != 0
        except:
            return False

def main():
    """Main application entry point"""
    setup_logging()
    logging.info("Starting Whisper Transcriber")
    
    requirements_met, warnings = check_system_requirements()
    if not requirements_met:
        warning_message = "\n".join(warnings)
        if not messagebox.askokcancel("System Requirements", 
                                    f"Warning: System requirements not met:\n\n{warning_message}\n\n"
                                    "Application may run slowly. Continue anyway?"):
            sys.exit(1)
    
    cleanup_temp_files()
    
    try:

        
        # More resilient admin check
        if sys.platform == 'win32':
            try:
                if not is_admin():
                    logging.warning("Running without admin privileges - some features may be limited")
            except Exception as e:
                logging.warning(f"Could not check admin status: {e}")
    except Exception as e:
        logging.warning(f"Error checking platform/admin status: {e}")
    
    try:
        app = WhisperTranscriberApp()
        app.run()
    except Exception as e:
        logging.critical(f"Fatal error: {e}", exc_info=True)
        messagebox.showerror("Fatal Error", 
                           f"A fatal error occurred:\n\n{str(e)}\n\n"
                           "Please check the logs for more information.")
        sys.exit(1)
    finally:
        logging.info("Application shutting down")

if __name__ == "__main__":
    main()
