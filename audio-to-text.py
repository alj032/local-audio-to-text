import tkinter as tk
from tkinter import ttk, messagebox
import sounddevice as sd
import numpy as np
import faster_whisper  # Changed to faster_whisper
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

class WhisperTranscriberApp:
    def __init__(self):
        # Define default config first
        self.default_config = {
            'start_hotkey': 'ctrl+shift+r',
            'stop_hotkey': 'ctrl+shift+s'
        }
        
        # Set up config file path
        self.config_file = Path.home() / '.whisper_transcriber_config.json'
        
        # Now load config (which uses default_config if needed)
        self.load_config()
        
        # Continue with rest of initialization
        self.setup_variables()
        self.create_tray_icon()
        self.create_window()
        self.setup_hotkeys()
        
    def setup_variables(self):
        # Initialize variables
        self.recording = False
        self.audio_queue = queue.Queue()
        self.recording_thread = None
        self.transcription_thread = None
        self.audio_data = []
        self.sample_rate = 16000
        self.window = None
        self.animation = None
        self.line = None
        self.stream = None  # Add stream variable to track active audio stream
        
        # Audio visualization variables
        self.buffer_size = 2000
        self.audio_data_display = np.zeros(self.buffer_size)
        self.plot_data = np.zeros(self.buffer_size)
        
        # Model settings - simplified for faster-whisper
        self.models = {
            "Tiny (fast, less accurate)": "tiny",
            "Base (balanced)": "base",
            "Small (accurate, slower)": "small"
        }
        self.current_model = None
        self.model = None
        
        # Get available audio devices
        self.audio_devices = self.get_audio_devices()

    def load_config(self):
        try:
            with open(self.config_file, 'r') as f:
                self.config = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            self.config = self.default_config.copy()
            self.save_config()
            
    def save_config(self):
        with open(self.config_file, 'w') as f:
            json.dump(self.config, f)
            
    def setup_hotkeys(self):
        # Remove any existing hotkeys
        try:
            keyboard.unhook_all()
        except:
            pass
            
        # Set up new hotkeys
        keyboard.add_hotkey(self.config['start_hotkey'], self.start_recording_hotkey)
        keyboard.add_hotkey(self.config['stop_hotkey'], self.stop_recording_hotkey)
        
    def start_recording_hotkey(self):
        if not self.recording:
            # Need to use after() to run in main thread
            self.window.after(0, self.start_recording)
            
    def stop_recording_hotkey(self):
        if self.recording:
            self.window.after(0, self.stop_recording)

    def get_audio_devices(self):
        devices = sd.query_devices()
        input_devices = {}
        for i, device in enumerate(devices):
            if device['max_input_channels'] > 0:
                input_devices[f"{device['name']}"] = i
        return input_devices

    def create_tray_icon(self):
        icon_size = 64
        icon_image = Image.new('RGB', (icon_size, icon_size), color='white')
        draw = ImageDraw.Draw(icon_image)
        draw.ellipse([4, 4, icon_size-4, icon_size-4], fill='blue')
        
        menu = (
            pystray.MenuItem("Show", self.show_window),
            pystray.MenuItem("Exit", self.quit_application)
        )
        
        self.tray_icon = pystray.Icon("whisper_transcriber", icon_image, "Whisper Transcriber", menu)

    def create_window(self):
        self.window = tk.Tk()
        self.window.title("Whisper Transcriber")
        self.window.protocol('WM_DELETE_WINDOW', self.hide_window)
        
        self.setup_ui()
    
    def setup_ui(self):
        main_frame = ttk.Frame(self.window, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Model selection
        ttk.Label(main_frame, text="Select Model:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.model_var = tk.StringVar(value="Tiny (fast, less accurate)")
        model_dropdown = ttk.Combobox(main_frame, textvariable=self.model_var, values=list(self.models.keys()))
        model_dropdown.grid(row=0, column=1, sticky=(tk.W, tk.E), pady=5)
        model_dropdown.bind('<<ComboboxSelected>>', self.on_model_change)
        
        # Audio device selection
        ttk.Label(main_frame, text="Audio Input:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.device_var = tk.StringVar(value=list(self.audio_devices.keys())[0] if self.audio_devices else "No devices found")
        device_dropdown = ttk.Combobox(main_frame, textvariable=self.device_var, values=list(self.audio_devices.keys()))
        device_dropdown.grid(row=1, column=1, sticky=(tk.W, tk.E), pady=5)
        
        # Auto-copy checkbox
        self.auto_copy_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(main_frame, text="Auto-copy to clipboard", variable=self.auto_copy_var).grid(row=2, column=0, columnspan=2, sticky=tk.W, pady=5)
        
        # Audio visualization
        self.setup_audio_visualization(main_frame)
        
        # Record button
        self.record_button = ttk.Button(main_frame, text="Start Recording", command=self.toggle_recording)
        self.record_button.grid(row=4, column=0, columnspan=2, pady=10)
        
        # Status label
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(main_frame, textvariable=self.status_var).grid(row=5, column=0, columnspan=2, pady=5)
        
        # Transcription text
        self.text_output = tk.Text(main_frame, height=10, width=50, wrap=tk.WORD)
        self.text_output.grid(row=6, column=0, columnspan=2, pady=5)
        
        # Copy button
        self.copy_button = ttk.Button(main_frame, text="Copy to Clipboard", command=self.copy_to_clipboard)
        self.copy_button.grid(row=7, column=0, columnspan=2, pady=5)
        
        # Load initial model
        self.on_model_change(None)

        # Add Hotkeys section
        hotkeys_frame = ttk.LabelFrame(main_frame, text="Hotkeys", padding="5")
        hotkeys_frame.grid(row=8, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)
        
        # Start hotkey
        ttk.Label(hotkeys_frame, text="Start Recording:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.start_hotkey_var = tk.StringVar(value=self.config['start_hotkey'])
        ttk.Label(hotkeys_frame, textvariable=self.start_hotkey_var).grid(row=0, column=1, sticky=tk.W, pady=2)
        ttk.Button(hotkeys_frame, text="Set", 
                  command=lambda: self.set_hotkey('start_hotkey')).grid(row=0, column=2, padx=5)
        
        # Stop hotkey
        ttk.Label(hotkeys_frame, text="Stop Recording:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.stop_hotkey_var = tk.StringVar(value=self.config['stop_hotkey'])
        ttk.Label(hotkeys_frame, textvariable=self.stop_hotkey_var).grid(row=1, column=1, sticky=tk.W, pady=2)
        ttk.Button(hotkeys_frame, text="Set", 
                  command=lambda: self.set_hotkey('stop_hotkey')).grid(row=1, column=2, padx=5)
        
        # Reset hotkeys button
        ttk.Button(hotkeys_frame, text="Reset to Defaults", 
                  command=self.reset_hotkeys).grid(row=2, column=0, columnspan=3, pady=5)
    
    def set_hotkey(self, hotkey_type):
        current = self.config[hotkey_type]
        
        def save_hotkey(new_hotkey):
            if new_hotkey != current:
                self.config[hotkey_type] = new_hotkey
                if hotkey_type == 'start_hotkey':
                    self.start_hotkey_var.set(new_hotkey)
                else:
                    self.stop_hotkey_var.set(new_hotkey)
                self.save_config()
                self.setup_hotkeys()
        
        HotkeyDialog(self.window, current, save_hotkey)
    
    def reset_hotkeys(self):
        if messagebox.askyesno("Reset Hotkeys", "Reset hotkeys to default values?"):
            self.config['start_hotkey'] = self.default_config['start_hotkey']
            self.config['stop_hotkey'] = self.default_config['stop_hotkey']
            self.start_hotkey_var.set(self.config['start_hotkey'])
            self.stop_hotkey_var.set(self.config['stop_hotkey'])
            self.save_config()
            self.setup_hotkeys()
    
    def setup_audio_visualization(self, parent):
        self.fig, self.ax = plt.subplots(figsize=(6, 2))
        self.canvas = FigureCanvasTkAgg(self.fig, master=parent)
        self.canvas.get_tk_widget().grid(row=3, column=0, columnspan=2, pady=5)
        
        x = np.linspace(0, self.buffer_size, self.buffer_size)
        self.line, = self.ax.plot(x, self.plot_data)
        
        self.ax.set_ylim(-1, 1)
        self.ax.set_xlim(0, self.buffer_size)
        self.ax.set_title('Audio Input')
        self.ax.grid(True)
        
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        
        self.fig.tight_layout()
    
    def update_audio_visualization(self, frame):
        if self.recording:
            try:
                # Get all available audio data
                data = []
                while not self.audio_queue.empty():
                    data.append(self.audio_queue.get_nowait())
                
                if data:
                    # Combine and normalize the new data
                    new_data = np.concatenate(data).flatten()
                    max_val = np.abs(new_data).max()
                    if max_val > 0:
                        new_data = new_data / max_val
                    
                    # Update the display buffer
                    data_len = len(new_data)
                    self.plot_data = np.roll(self.plot_data, -data_len)
                    self.plot_data[-data_len:] = new_data
                    
                    # Update the line data
                    self.line.set_ydata(self.plot_data)
                    
                    # Force a redraw
                    self.fig.canvas.draw_idle()
                    self.fig.canvas.flush_events()
            
            except queue.Empty:
                pass
        
        return self.line,
    
    def on_model_change(self, event):
        model_name = self.models[self.model_var.get()]
        if model_name != self.current_model:
            self.status_var.set("Loading model...")
            self.window.update()
            
            def load_model():
                try:
                    # Using faster-whisper which handles model downloads better
                    self.model = faster_whisper.WhisperModel(model_name, device="cpu", compute_type="int8")
                    self.current_model = model_name
                    self.status_var.set("Model loaded successfully")
                except Exception as e:
                    self.status_var.set(f"Error loading model: {str(e)}")
            
            threading.Thread(target=load_model).start()

    def toggle_recording(self):
        if not self.recording:
            self.start_recording()
        else:
            self.stop_recording()
    
    def start_recording(self):
        if not self.model:
            self.status_var.set("Please wait for model to load")
            return
            
        if self.recording:
            return  # Prevent multiple recordings
            
        # Clear previous recording data
        self.audio_data = []
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break

        self.recording = True
        self.record_button.config(text="Stop Recording")
        self.status_var.set("Recording...")
        
        # Reset visualization
        self.plot_data = np.zeros(self.buffer_size)
        self.line.set_ydata(self.plot_data)
        
        # Start visualization update
        self.animation = animation.FuncAnimation(
            self.fig,
            self.update_audio_visualization,
            interval=30,
            blit=True,
            cache_frame_data=False
        )
        
        # Start recording
        self.recording_thread = threading.Thread(target=self.record_audio)
        self.recording_thread.start()
    
    def stop_recording(self):
        if not self.recording:
            return

        self.recording = False
        self.record_button.config(text="Start Recording")
        self.status_var.set("Processing audio...")
        
        # Stop the animation
        if self.animation:
            self.animation.event_source.stop()
            self.animation = None
        
        # Close the audio stream
        if self.stream:
            self.stream.close()
            self.stream = None
        
        # Wait for recording thread to finish
        if self.recording_thread and self.recording_thread.is_alive():
            self.recording_thread.join()
        
        # Start transcription in a new thread
        self.transcription_thread = threading.Thread(target=self.process_audio)
        self.transcription_thread.start()
    
    def record_audio(self):
        def audio_callback(indata, frames, time, status):
            if status:
                print(status)
            if self.recording:  # Only add data if we're still recording
                self.audio_queue.put(indata.copy())
                self.audio_data.append(indata.copy())

        try:
            device_idx = self.audio_devices[self.device_var.get()]
            self.stream = sd.InputStream(
                callback=audio_callback,
                channels=1,
                samplerate=self.sample_rate,
                device=device_idx,
                blocksize=1024
            )
            
            with self.stream:
                while self.recording:
                    sd.sleep(100)  # Sleep to prevent busy-waiting
                    
        except Exception as e:
            self.status_var.set(f"Recording error: {str(e)}")
            self.recording = False
        finally:
            if self.stream:
                self.stream.close()
                self.stream = None


    def process_audio(self):
        try:
            # Check if we have any audio data
            if not self.audio_data or len(self.audio_data) == 0:
                self.status_var.set("No audio data recorded")
                return

            # Safely concatenate audio data
            try:
                audio = np.concatenate(self.audio_data)
            except ValueError as e:
                self.status_var.set("Error: No audio data to process")
                return
            
            # Check if audio data is empty or too short
            if len(audio) < 100:  # arbitrary minimum length
                self.status_var.set("Recording too short")
                return

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                sf.write(temp_file.name, audio, self.sample_rate)
                
                # Check if model is loaded
                if self.model is None:
                    self.status_var.set("Error: Model not loaded")
                    return

                # Using faster-whisper's transcribe method
                segments, _ = self.model.transcribe(temp_file.name)
                text = " ".join([segment.text for segment in segments]).strip()
                
                self.text_output.delete(1.0, tk.END)
                self.text_output.insert(tk.END, text)
                
                if self.auto_copy_var.get():
                    self.copy_to_clipboard()
                
                self.status_var.set("Transcription complete")
            
            Path(temp_file.name).unlink()
            
        except Exception as e:
            self.status_var.set(f"Error during transcription: {str(e)}")
        finally:
            # Clear audio data after processing
            self.audio_data = []

    def copy_to_clipboard(self):
        text = self.text_output.get(1.0, tk.END).strip()
        if text:
            pyperclip.copy(text)
            self.status_var.set("Copied to clipboard")
        else:
            self.status_var.set("No text to copy")
    
    def show_window(self):
        self.window.deiconify()
    
    def hide_window(self):
        self.window.withdraw()
    
    def quit_application(self):
            self.recording = False
            
            # Stop and clean up the audio stream
            if self.stream:
                self.stream.close()
                self.stream = None
                
            # Unhook keyboard listeners
            try:
                keyboard.unhook_all()
            except:
                pass
            
            # Wait for recording thread
            if self.recording_thread and self.recording_thread.is_alive():
                self.recording_thread.join()
                
            self.window.quit()
            self.tray_icon.stop()
    
    def run(self):
        self.window.withdraw()
        tray_thread = threading.Thread(target=self.tray_icon.run)
        tray_thread.daemon = True
        tray_thread.start()
        self.window.mainloop()

class HotkeyDialog(tk.Toplevel):
    def __init__(self, parent, current_hotkey, on_save):
        super().__init__(parent)
        self.title("Set Hotkey")
        self.resizable(False, False)
        
        self.current_hotkey = current_hotkey
        self.on_save = on_save
        self.pressed_keys = set()
        
        self.setup_ui()
        
        # Make dialog modal
        self.transient(parent)
        self.grab_set()
        
    def setup_ui(self):
        frame = ttk.Frame(self, padding="10")
        frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        ttk.Label(frame, text="Press the desired key combination:").grid(row=0, column=0, columnspan=2, pady=5)
        
        self.hotkey_var = tk.StringVar(value=self.current_hotkey)
        self.hotkey_entry = ttk.Entry(frame, textvariable=self.hotkey_var, state='readonly')
        self.hotkey_entry.grid(row=1, column=0, columnspan=2, pady=5)
        
        ttk.Button(frame, text="Clear", command=self.clear_hotkey).grid(row=2, column=0, pady=10)
        ttk.Button(frame, text="Save", command=self.save_hotkey).grid(row=2, column=1, pady=10)
        
        # Bind both key press and release events
        self.bind('<KeyPress>', self.on_key_press)
        self.bind('<KeyRelease>', self.on_key_release)
        
    def clear_hotkey(self):
        self.hotkey_var.set("")
        self.pressed_keys.clear()
        
    def save_hotkey(self):
        hotkey = self.hotkey_var.get()
        if hotkey:
            self.on_save(hotkey)
            self.destroy()
    
    def on_key_press(self, event):
        # Ignore special events
        if event.keysym in ('Escape', 'Return', 'Tab'):
            return
            
        # Convert keysym to lowercase for consistency
        key = event.keysym.lower()
        
        # Map modifier keys to their standard names
        modifier_map = {
            'control_l': 'ctrl',
            'control_r': 'ctrl',
            'shift_l': 'shift',
            'shift_r': 'shift',
            'alt_l': 'alt',
            'alt_r': 'alt'
        }
        
        # Convert modifier keys to their standard names
        if key in modifier_map:
            key = modifier_map[key]
            
        # Add the key to pressed keys
        self.pressed_keys.add(key)
        
        # Update the hotkey display
        self.update_hotkey_display()
    
    def on_key_release(self, event):
        key = event.keysym.lower()
        
        # Map modifier keys to their standard names
        modifier_map = {
            'control_l': 'ctrl',
            'control_r': 'ctrl',
            'shift_l': 'shift',
            'shift_r': 'shift',
            'alt_l': 'alt',
            'alt_r': 'alt'
        }
        
        if key in modifier_map:
            key = modifier_map[key]
            
        # Remove the key from pressed keys
        self.pressed_keys.discard(key)
        
        # Update the hotkey display
        self.update_hotkey_display()
    
    def update_hotkey_display(self):
        # Sort keys to ensure consistent order (modifiers first)
        modifier_order = {'ctrl': 0, 'shift': 1, 'alt': 2}
        
        # Separate modifiers and regular keys
        modifiers = sorted(
            [k for k in self.pressed_keys if k in modifier_order],
            key=lambda x: modifier_order[x]
        )
        other_keys = [k for k in self.pressed_keys if k not in modifier_order]
        
        # Combine all keys in the correct order
        all_keys = modifiers + other_keys
        
        if all_keys:
            # Join all keys with '+'
            hotkey = '+'.join(all_keys)
            self.hotkey_var.set(hotkey)

def main():
    app = WhisperTranscriberApp()
    app.run()

if __name__ == "__main__":
    main()
