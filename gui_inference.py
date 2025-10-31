"""
Real-time keyword spotting GUI
Continuously streams audio and displays detection results
"""

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import torch
import torch.nn.functional as F
import numpy as np
import threading
import time
import queue
from pathlib import Path

from model import TinyKWS, MinimalKWS
from model_2d import build_model
from audio_utils import AudioPreprocessor, AudioStreamer
from temporal_smoother import TemporalSmoother, AdvancedSmoother
from config import ProjectConfig
from inference_enhanced import SlidingWindowDetector
from collections import deque


class KeywordSpottingGUI:
    """GUI application for real-time keyword spotting"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Real-Time Keyword Spotting - Hello Detector")
        self.root.geometry("800x600")
        self.root.configure(bg='#2b2b2b')
        
        # State variables
        self.model = None
        self.model_path = None
        self.preprocessor = None
        self.streamer = None
        self.is_streaming = False
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.threshold = 0.7
        self.model_type = "tiny"
        
        # Config
        self.config = ProjectConfig()
        
        # Enhanced inference
        self.use_enhanced_inference = True  # Use sliding window by default
        self.detector = None
        
        # Sliding window averaging (like Hello repo)
        self.detection_scores = deque()
        self.last_detection_time = 0.0
        self.buffer = None  # Rolling audio buffer
        
        # Temporal smoothing (fallback if enhanced inference disabled)
        self.use_smoothing = True
        self.smoother = None
        
        # Audio queue for thread-safe communication
        self.audio_queue = queue.Queue()
        self.result_queue = queue.Queue()
        
        # Statistics
        self.stats = {
            'total_chunks': 0,
            'detections': 0,
            'avg_latency': 0.0,
            'last_detection_time': None
        }
        
        self.setup_gui()
        
    def setup_gui(self):
        """Create GUI components"""
        # Style configuration
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('Title.TLabel', font=('Arial', 16, 'bold'), background='#2b2b2b', foreground='white')
        style.configure('Header.TLabel', font=('Arial', 12, 'bold'), background='#2b2b2b', foreground='#4CAF50')
        style.configure('Status.TLabel', font=('Arial', 11), background='#2b2b2b', foreground='white')
        
        # Main container
        main_frame = tk.Frame(self.root, bg='#2b2b2b', padx=20, pady=20)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        title_label = tk.Label(
            main_frame, 
            text="🎤 Real-Time Hello Detector", 
            font=('Arial', 20, 'bold'),
            bg='#2b2b2b',
            fg='#4CAF50'
        )
        title_label.pack(pady=(0, 20))
        
        # Model selection frame
        model_frame = tk.LabelFrame(
            main_frame,
            text="Model Configuration",
            font=('Arial', 11, 'bold'),
            bg='#3b3b3b',
            fg='white',
            padx=15,
            pady=10
        )
        model_frame.pack(fill=tk.X, pady=(0, 15))
        
        # Model path selection
        path_frame = tk.Frame(model_frame, bg='#3b3b3b')
        path_frame.pack(fill=tk.X, pady=(0, 10))
        
        tk.Label(
            path_frame,
            text="Model Path:",
            bg='#3b3b3b',
            fg='white',
            font=('Arial', 10)
        ).pack(side=tk.LEFT, padx=(0, 10))
        
        self.model_path_label = tk.Label(
            path_frame,
            text="No model loaded",
            bg='#3b3b3b',
            fg='#888888',
            font=('Arial', 9),
            anchor='w'
        )
        self.model_path_label.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        self.load_model_btn = tk.Button(
            path_frame,
            text="Load Model",
            command=self.load_model,
            bg='#2196F3',
            fg='white',
            font=('Arial', 10),
            padx=15,
            pady=5,
            relief=tk.FLAT,
            cursor='hand2'
        )
        self.load_model_btn.pack(side=tk.RIGHT, padx=(10, 0))
        
        # Threshold control
        threshold_frame = tk.Frame(model_frame, bg='#3b3b3b')
        threshold_frame.pack(fill=tk.X)
        
        tk.Label(
            threshold_frame,
            text="Confidence Threshold:",
            bg='#3b3b3b',
            fg='white',
            font=('Arial', 10)
        ).pack(side=tk.LEFT, padx=(0, 10))
        
        self.threshold_var = tk.DoubleVar(value=0.7)
        self.threshold_scale = tk.Scale(
            threshold_frame,
            from_=0.1,
            to=0.99,
            resolution=0.01,
            orient=tk.HORIZONTAL,
            variable=self.threshold_var,
            bg='#3b3b3b',
            fg='white',
            highlightthickness=0,
            length=200,
            command=self.update_threshold
        )
        self.threshold_scale.pack(side=tk.LEFT, padx=(0, 10))
        
        self.threshold_label = tk.Label(
            threshold_frame,
            text="0.70",
            bg='#3b3b3b',
            fg='white',
            font=('Arial', 10),
            width=5
        )
        self.threshold_label.pack(side=tk.LEFT)
        
        # Enhanced inference control
        enhanced_frame = tk.Frame(model_frame, bg='#3b3b3b')
        enhanced_frame.pack(fill=tk.X, pady=(10, 0))
        
        self.enhanced_var = tk.BooleanVar(value=True)
        enhanced_check = tk.Checkbutton(
            enhanced_frame,
            text="Use Sliding Window Averaging (Reduces False Positives)",
            variable=self.enhanced_var,
            bg='#3b3b3b',
            fg='white',
            selectcolor='#2b2b2b',
            font=('Arial', 10),
            command=self.toggle_enhanced
        )
        enhanced_check.pack(side=tk.LEFT)
        
        # Averaging window control
        avg_frame = tk.Frame(model_frame, bg='#3b3b3b')
        avg_frame.pack(fill=tk.X, pady=(5, 0))
        
        tk.Label(
            avg_frame,
            text="Averaging Window (ms):",
            bg='#3b3b3b',
            fg='white',
            font=('Arial', 9)
        ).pack(side=tk.LEFT, padx=(0, 10))
        
        self.avg_window_var = tk.IntVar(value=self.config.inference.averaging_window_ms)
        avg_spin = tk.Spinbox(
            avg_frame,
            from_=100,
            to=3000,
            textvariable=self.avg_window_var,
            width=6,
            bg='#4b4b4b',
            fg='white',
            font=('Arial', 9),
            increment=100
        )
        avg_spin.pack(side=tk.LEFT, padx=(0, 10))
        
        # Smoothing control (fallback)
        smoothing_frame = tk.Frame(model_frame, bg='#3b3b3b')
        smoothing_frame.pack(fill=tk.X, pady=(5, 0))
        
        self.smoothing_var = tk.BooleanVar(value=True)
        smoothing_check = tk.Checkbutton(
            smoothing_frame,
            text="Enable Temporal Smoothing (fallback if enhanced disabled)",
            variable=self.smoothing_var,
            bg='#3b3b3b',
            fg='white',
            selectcolor='#2b2b2b',
            font=('Arial', 9),
            command=self.toggle_smoothing
        )
        smoothing_check.pack(side=tk.LEFT)
        
        # Smoothing window size
        window_frame = tk.Frame(model_frame, bg='#3b3b3b')
        window_frame.pack(fill=tk.X, pady=(5, 0))
        
        tk.Label(
            window_frame,
            text="Smoothing Window:",
            bg='#3b3b3b',
            fg='white',
            font=('Arial', 9)
        ).pack(side=tk.LEFT, padx=(0, 10))
        
        self.window_size_var = tk.IntVar(value=3)
        window_spin = tk.Spinbox(
            window_frame,
            from_=1,
            to=10,
            textvariable=self.window_size_var,
            width=5,
            bg='#4b4b4b',
            fg='white',
            font=('Arial', 9),
            command=self.update_smoother
        )
        window_spin.pack(side=tk.LEFT, padx=(0, 10))
        
        tk.Label(
            window_frame,
            text="(higher = fewer false positives, more latency)",
            bg='#3b3b3b',
            fg='#888888',
            font=('Arial', 8)
        ).pack(side=tk.LEFT)
        
        # Detection display frame
        detection_frame = tk.LabelFrame(
            main_frame,
            text="Live Detection",
            font=('Arial', 11, 'bold'),
            bg='#3b3b3b',
            fg='white',
            padx=15,
            pady=15
        )
        detection_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 15))
        
        # Status indicator
        status_container = tk.Frame(detection_frame, bg='#3b3b3b')
        status_container.pack(fill=tk.X, pady=(0, 20))
        
        self.status_canvas = tk.Canvas(
            status_container,
            width=100,
            height=100,
            bg='#3b3b3b',
            highlightthickness=0
        )
        self.status_canvas.pack(side=tk.LEFT, padx=(0, 20))
        
        # Draw initial status circle (gray)
        self.status_circle = self.status_canvas.create_oval(
            10, 10, 90, 90,
            fill='#666666',
            outline='#888888',
            width=2
        )
        
        # Detection text
        text_frame = tk.Frame(detection_frame, bg='#3b3b3b')
        text_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.detection_label = tk.Label(
            text_frame,
            text="Waiting to start...",
            font=('Arial', 24, 'bold'),
            bg='#3b3b3b',
            fg='#888888'
        )
        self.detection_label.pack(pady=(0, 10))
        
        self.confidence_label = tk.Label(
            text_frame,
            text="",
            font=('Arial', 14),
            bg='#3b3b3b',
            fg='#aaaaaa'
        )
        self.confidence_label.pack()
        
        self.latency_label = tk.Label(
            text_frame,
            text="",
            font=('Arial', 11),
            bg='#3b3b3b',
            fg='#888888'
        )
        self.latency_label.pack(pady=(10, 0))
        
        # Statistics frame
        stats_frame = tk.LabelFrame(
            main_frame,
            text="Statistics",
            font=('Arial', 11, 'bold'),
            bg='#3b3b3b',
            fg='white',
            padx=15,
            pady=10
        )
        stats_frame.pack(fill=tk.X, pady=(0, 15))
        
        self.stats_label = tk.Label(
            stats_frame,
            text="Total Chunks: 0  |  Detections: 0  |  Avg Latency: 0.0ms",
            font=('Arial', 10),
            bg='#3b3b3b',
            fg='white',
            anchor='w'
        )
        self.stats_label.pack(fill=tk.X)
        
        # Control buttons
        control_frame = tk.Frame(main_frame, bg='#2b2b2b')
        control_frame.pack(fill=tk.X)
        
        self.start_btn = tk.Button(
            control_frame,
            text="▶ Start Streaming",
            command=self.start_streaming,
            bg='#4CAF50',
            fg='white',
            font=('Arial', 12, 'bold'),
            padx=30,
            pady=10,
            relief=tk.FLAT,
            cursor='hand2',
            state=tk.DISABLED
        )
        self.start_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        self.stop_btn = tk.Button(
            control_frame,
            text="⏹ Stop Streaming",
            command=self.stop_streaming,
            bg='#f44336',
            fg='white',
            font=('Arial', 12, 'bold'),
            padx=30,
            pady=10,
            relief=tk.FLAT,
            cursor='hand2',
            state=tk.DISABLED
        )
        self.stop_btn.pack(side=tk.LEFT)
        
        # Initialize preprocessor
        self.preprocessor = AudioPreprocessor(sample_rate=16000)
        
        # Start GUI update loop
        self.update_gui()
        
    def load_model(self):
        """Load model from file"""
        file_path = filedialog.askopenfilename(
            title="Select Model File",
            filetypes=[("PyTorch Models", "*.pth *.pt"), ("TorchScript", "*.ts"), ("All Files", "*.*")]
        )
        
        if not file_path:
            return
        
        try:
            self.root.config(cursor="wait")
            self.root.update()
            
            print(f"Loading model from {file_path}...")
            
            # Try TorchScript first
            if file_path.endswith('.ts'):
                try:
                    self.model = torch.jit.load(file_path, map_location=self.device).eval()
                    self.model_path = file_path
                    self.model_path_label.config(
                        text=Path(file_path).name + " (TorchScript)",
                        fg='#4CAF50'
                    )
                    self.start_btn.config(state=tk.NORMAL)
                    messagebox.showinfo("Success", "TorchScript model loaded!")
                    return
                except Exception as e:
                    print(f"TorchScript load failed: {e}, trying PyTorch format")
            
            # Load PyTorch checkpoint
            checkpoint = torch.load(file_path, map_location=self.device, weights_only=False)
            
            # Get config if available
            if "config" in checkpoint:
                self.config = checkpoint["config"]
                if hasattr(self.config.model, "num_classes"):
                    self.config.model.num_classes = checkpoint.get("num_classes", 3)
            
            # Determine model type and architecture
            use_2d = checkpoint.get("use_2d_conv", False)
            self.model_type = checkpoint.get("model_type", "tiny").replace("_quantized", "")
            num_classes = checkpoint.get("num_classes", 3)
            
            # Create model
            if use_2d:
                from model_2d import build_model
                self.config.model.use_2d_conv = True
                self.config.model.model_type = self.model_type
                self.config.model.num_classes = num_classes
                self.model = build_model(self.config.model)
            else:
                if self.model_type == "tiny":
                    self.model = TinyKWS(num_classes=num_classes, num_filters=64)
                else:
                    self.model = MinimalKWS(num_classes=num_classes, num_filters=32)
            
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.model = self.model.to(self.device)
            self.model.eval()
            
            # Create detector if enhanced inference enabled
            if self.use_enhanced_inference:
                self.detector = SlidingWindowDetector(self.model, self.config)
            
            self.model_path = file_path
            self.model_path_label.config(
                text=Path(file_path).name,
                fg='#4CAF50'
            )
            self.start_btn.config(state=tk.NORMAL)
            
            # Update threshold from config
            if hasattr(self.config, "inference"):
                self.threshold = self.config.inference.detection_threshold
                self.threshold_var.set(self.threshold)
                self.threshold_label.config(text=f"{self.threshold:.2f}")
                if self.config.inference.averaging_window_ms:
                    self.avg_window_var.set(self.config.inference.averaging_window_ms)
            
            messagebox.showinfo("Success", f"Model loaded successfully!\nClasses: {num_classes}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model:\n{str(e)}")
            import traceback
            traceback.print_exc()
        finally:
            self.root.config(cursor="")
            self.root.update()
    
    def update_threshold(self, value=None):
        """Update threshold value"""
        self.threshold = self.threshold_var.get()
        self.threshold_label.config(text=f"{self.threshold:.2f}")
        # Update config
        self.config.inference.detection_threshold = self.threshold
        # Update smoother threshold
        if self.smoother:
            self.smoother.min_confidence = self.threshold
    
    def toggle_enhanced(self):
        """Toggle enhanced sliding window inference"""
        self.use_enhanced_inference = self.enhanced_var.get()
        if self.use_enhanced_inference:
            self.config.inference.averaging_window_ms = self.avg_window_var.get()
    
    def toggle_smoothing(self):
        """Toggle temporal smoothing"""
        self.use_smoothing = self.smoothing_var.get()
        if self.use_smoothing:
            self.update_smoother()
        else:
            self.smoother = None
    
    def update_smoother(self):
        """Update smoother configuration"""
        if self.use_smoothing:
            window_size = self.window_size_var.get()
            self.smoother = AdvancedSmoother(
                window_size=window_size,
                min_confidence=self.threshold,
                debounce_ms=800,
                adaptive_threshold=True
            )
    
    def start_streaming(self):
        """Start audio streaming"""
        if self.model is None:
            messagebox.showerror("Error", "Please load a model first!")
            return
        
        try:
            self.is_streaming = True
            self.start_btn.config(state=tk.DISABLED)
            self.stop_btn.config(state=tk.NORMAL)
            
            # Reset statistics
            self.stats = {
                'total_chunks': 0,
                'detections': 0,
                'avg_latency': 0.0,
                'last_detection_time': None,
                'false_positives': 0
            }
            
            # Initialize rolling buffer for enhanced inference
            if self.use_enhanced_inference:
                self.buffer = np.zeros(
                    self.config.audio.target_num_samples, 
                    dtype=np.float32
                )
                self.detection_scores.clear()
                self.last_detection_time = 0.0
                if self.detector is None:
                    self.detector = SlidingWindowDetector(self.model, self.config)
                else:
                    self.detector.reset()
            
            # Initialize smoother if enabled (fallback)
            if not self.use_enhanced_inference and self.use_smoothing:
                self.update_smoother()
            else:
                self.smoother = None
            
            # Start audio streaming thread
            self.stream_thread = threading.Thread(target=self._stream_audio, daemon=True)
            self.stream_thread.start()
            
            # Start inference thread
            self.inference_thread = threading.Thread(target=self._inference_loop, daemon=True)
            self.inference_thread.start()
            
            self.detection_label.config(text="Streaming...", fg='#2196F3')
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start streaming:\n{str(e)}")
            self.stop_streaming()
    
    def stop_streaming(self):
        """Stop audio streaming"""
        self.is_streaming = False
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.detection_label.config(text="Stopped", fg='#888888')
        self.confidence_label.config(text="")
        self.latency_label.config(text="")
        self.update_status_circle('#666666')
    
    def _stream_audio(self):
        """Audio streaming thread"""
        try:
            import sounddevice as sd
            
            chunk_size = int(16000 * 0.5)  # 500ms chunks for low latency
            
            def audio_callback(indata, frames, time_info, status):
                if status:
                    print(f"Audio status: {status}")
                if self.is_streaming:
                    audio_chunk = indata[:, 0].flatten()
                    self.audio_queue.put(audio_chunk)
            
            with sd.InputStream(
                samplerate=16000,
                channels=1,
                callback=audio_callback,
                blocksize=chunk_size,
                dtype='float32'
            ):
                while self.is_streaming:
                    time.sleep(0.1)
                    
        except ImportError:
            self.result_queue.put(("error", "sounddevice not installed. Install with: pip install sounddevice"))
        except Exception as e:
            self.result_queue.put(("error", f"Audio streaming error: {str(e)}"))
    
    def _inference_loop(self):
        """Inference processing loop with enhanced sliding window support"""
        while self.is_streaming:
            try:
                # Get audio chunk (non-blocking with timeout)
                audio_chunk = self.audio_queue.get(timeout=0.1)
                
                start_time = time.time()
                
                if self.use_enhanced_inference and self.detector:
                    # Use enhanced sliding window inference
                    # Update rolling buffer
                    self.buffer = np.roll(self.buffer, -len(audio_chunk))
                    self.buffer[-len(audio_chunk):] = audio_chunk
                    
                    # Normalize
                    normalized = self.buffer / (np.linalg.norm(self.buffer) + 1e-6)
                    
                    # Detect with sliding window
                    is_detected, avg_score = self.detector.predict(normalized)
                    
                    inference_time = (time.time() - start_time) * 1000
                    total_time = inference_time
                    
                    is_hello = is_detected
                    conf = avg_score
                    raw_is_hello = is_detected  # For display
                    
                else:
                    # Standard inference with temporal smoothing fallback
                    # Preprocess
                    features = self.preprocessor.stream_preprocess(audio_chunk)
                    
                    # Inference
                    inference_start = time.time()
                    features = features.to(self.device)
                    
                    with torch.no_grad():
                        outputs = self.model(features)
                        probs = F.softmax(outputs, dim=1)
                        confidence, predicted = torch.max(probs, 1)
                        
                        # Handle 3-class vs 2-class
                        num_classes = outputs.shape[1]
                        if num_classes == 3:
                            hello_idx = self.config.labels.index("hello")
                            raw_is_hello = (predicted.item() == hello_idx) and (confidence.item() > self.threshold)
                            raw_conf = float(probs[0, hello_idx])
                        else:
                            # Binary classification (backward compatibility)
                            raw_is_hello = (predicted.item() == 1) and (confidence.item() > self.threshold)
                            raw_conf = confidence.item()
                    
                    inference_time = (time.time() - inference_start) * 1000
                    total_time = (time.time() - start_time) * 1000
                    
                    # Apply temporal smoothing if enabled
                    if self.smoother:
                        current_time = time.time()
                        is_hello, smoothed_conf = self.smoother.update(
                            raw_is_hello, raw_conf, current_time
                        )
                        conf = smoothed_conf
                    else:
                        is_hello = raw_is_hello
                        conf = raw_conf
                
                # Update statistics
                self.stats['total_chunks'] += 1
                if is_hello:
                    self.stats['detections'] += 1
                    self.stats['last_detection_time'] = time.time()
                elif raw_is_hello and not is_hello:
                    # Filtered out (potential false positive)
                    self.stats['false_positives'] = self.stats.get('false_positives', 0) + 1
                
                # Update average latency
                if self.stats['total_chunks'] == 1:
                    self.stats['avg_latency'] = total_time
                else:
                    self.stats['avg_latency'] = (
                        self.stats['avg_latency'] * (self.stats['total_chunks'] - 1) + total_time
                    ) / self.stats['total_chunks']
                
                # Send result to GUI thread
                self.result_queue.put((
                    "detection",
                    is_hello,
                    conf,
                    total_time,
                    inference_time,
                    raw_is_hello  # Also send raw detection
                ))
                
            except queue.Empty:
                continue
            except Exception as e:
                self.result_queue.put(("error", f"Inference error: {str(e)}"))
                import traceback
                traceback.print_exc()
    
    def update_status_circle(self, color):
        """Update status indicator circle"""
        self.status_canvas.delete(self.status_circle)
        self.status_circle = self.status_canvas.create_oval(
            10, 10, 90, 90,
            fill=color,
            outline=color,
            width=0
        )
    
    def update_gui(self):
        """Update GUI with latest results"""
        # Process all queued results
        while True:
            try:
                result = self.result_queue.get_nowait()
                
                if result[0] == "detection":
                    _, is_hello, conf, total_time, inference_time, raw_is_hello = result
                    
                    if is_hello:
                        self.detection_label.config(text="🎤 HELLO DETECTED!", fg='#4CAF50')
                        self.confidence_label.config(
                            text=f"Confidence: {conf:.2%}",
                            fg='#4CAF50'
                        )
                        self.update_status_circle('#4CAF50')
                    elif raw_is_hello and not is_hello:
                        # Filtered by smoother
                        self.detection_label.config(text="Filtering...", fg='#FF9800')
                        self.confidence_label.config(
                            text=f"Raw: {conf:.2%} (smoothed out)",
                            fg='#FF9800'
                        )
                        self.update_status_circle('#FF9800')
                    else:
                        self.detection_label.config(text="Listening...", fg='#888888')
                        self.confidence_label.config(
                            text=f"Confidence: {conf:.2%}",
                            fg='#888888'
                        )
                        self.update_status_circle('#666666')
                    
                    self.latency_label.config(
                        text=f"Latency: {inference_time:.1f}ms (inference) | {total_time:.1f}ms (total)",
                        fg='#aaaaaa'
                    )
                
                elif result[0] == "error":
                    messagebox.showerror("Error", result[1])
                    
            except queue.Empty:
                break
        
        # Update statistics
        false_positives = self.stats.get('false_positives', 0)
        stats_text = (
            f"Total Chunks: {self.stats['total_chunks']}  |  "
            f"Detections: {self.stats['detections']}  |  "
            f"Filtered FP: {false_positives}  |  "
            f"Avg Latency: {self.stats['avg_latency']:.1f}ms"
        )
        self.stats_label.config(text=stats_text)
        
        # Schedule next update
        self.root.after(50, self.update_gui)
    
    def on_closing(self):
        """Handle window closing"""
        self.stop_streaming()
        time.sleep(0.5)  # Give threads time to stop
        self.root.destroy()


def main():
    root = tk.Tk()
    app = KeywordSpottingGUI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()


if __name__ == "__main__":
    main()

