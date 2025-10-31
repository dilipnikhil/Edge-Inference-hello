"""
Enhanced real-time inference with sliding window averaging
Inspired by Hello repository - uses score averaging and suppression
"""

import torch
import torch.nn.functional as F
import numpy as np
import time
import queue
from collections import deque
from typing import Tuple

from config import ProjectConfig
from model_2d import build_model
from audio_utils import AudioPreprocessor


class SlidingWindowDetector:
    """
    Real-time detector with sliding window averaging
    Similar to Hello repository approach
    """
    
    def __init__(self, model, config: ProjectConfig):
        self.model = model
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Detection history: (timestamp, score)
        self.detection_scores: deque = deque()
        self.last_detection_time = 0.0
        
        # Preprocessor
        self.preprocessor = AudioPreprocessor(
            sample_rate=config.audio.sample_rate,
            n_mfcc=config.audio.n_mfcc
        )
        
    def predict(self, audio_chunk: np.ndarray) -> Tuple[bool, float]:
        """
        Predict with sliding window averaging
        
        Args:
            audio_chunk: Audio array (numpy)
        
        Returns:
            (is_detected, average_score) tuple
        """
        # Preprocess
        mfcc = self.preprocessor.extract_mfcc(audio_chunk, add_noise=False)
        if mfcc is None:
            return False, 0.0
        
        # Pad/truncate
        mfcc = self.preprocessor.pad_or_truncate(
            mfcc, 
            target_length=self.config.audio.target_num_samples // self.preprocessor.hop_length
        )
        
        # Convert to tensor and add dimensions for 2D model: (1, 1, n_mfcc, time_frames)
        if self.config.model.use_2d_conv:
            mfcc_tensor = torch.FloatTensor(mfcc.T).unsqueeze(0).unsqueeze(0)
        else:
            # 1D model expects (1, 1, time_frames) - use first MFCC coefficient
            mfcc_tensor = torch.FloatTensor(mfcc[:, 0]).unsqueeze(0).unsqueeze(0)
        
        mfcc_tensor = mfcc_tensor.to(self.device)
        
        # Inference
        self.model.eval()
        with torch.no_grad():
            logits = self.model(mfcc_tensor)
            probs = F.softmax(logits, dim=1)
            
            # Get hello class probability (index 2 in 3-class model)
            hello_idx = self.config.labels.index("hello")
            hello_prob = float(probs[0, hello_idx])
        
        # Add to detection history
        now = time.time()
        self.detection_scores.append((now, hello_prob))
        
        # Remove old scores outside averaging window
        cutoff = now - self.config.inference.averaging_window_ms / 1000
        while self.detection_scores and self.detection_scores[0][0] < cutoff:
            self.detection_scores.popleft()
        
        # Calculate average score
        if len(self.detection_scores) == 0:
            return False, 0.0
        
        avg_score = sum(score for _, score in self.detection_scores) / len(self.detection_scores)
        
        # Check threshold and suppression
        time_since_last = now - self.last_detection_time
        is_detected = (
            avg_score >= self.config.inference.detection_threshold and
            time_since_last >= self.config.inference.suppression_ms / 1000
        )
        
        if is_detected:
            self.last_detection_time = now
            # Clear recent scores to prevent immediate re-trigger
            self.detection_scores.clear()
        
        return is_detected, avg_score
    
    def reset(self):
        """Reset detector state"""
        self.detection_scores.clear()
        self.last_detection_time = 0.0


class TorchScriptRunner:
    """Runner for TorchScript models (like Hello repo)"""
    
    def __init__(self, model_path: str, device: str = "cpu"):
        self.model = torch.jit.load(model_path, map_location=device).eval()
        self.device = device
    
    def __call__(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Run inference and return probabilities"""
        with torch.inference_mode():
            logits = self.model(input_tensor)
            return torch.softmax(logits, dim=1)


def stream_detection(model_path: str, config: ProjectConfig = None):
    """
    Real-time streaming detection with sliding window
    """
    import sounddevice as sd
    import sys
    
    if config is None:
        config = ProjectConfig()
    
    # Load model
    try:
        # Try TorchScript first
        runner = TorchScriptRunner(model_path)
        use_torchscript = True
    except:
        # Fallback to regular PyTorch model
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        model = build_model(config.model)
        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.eval()
        runner = None
        use_torchscript = False
    
    detector = SlidingWindowDetector(model if not use_torchscript else None, config)
    
    # Audio buffer and queue
    buffer = np.zeros(config.audio.target_num_samples, dtype=np.float32)
    audio_queue = queue.Queue()
    
    def audio_callback(indata, frames, time_info, status):
        if status:
            print(status, file=sys.stderr)
        mono = indata.mean(axis=1).astype(np.float32)
        audio_queue.put(mono)
    
    window_samples = int(config.audio.sample_rate * config.inference.window_ms / 1000)
    
    with sd.InputStream(
        samplerate=config.audio.sample_rate,
        channels=1,
        blocksize=window_samples,
        callback=audio_callback,
    ):
        print("Listening for 'hello'... Press Ctrl+C to stop.")
        print(f"Threshold: {config.inference.detection_threshold:.2f}")
        print(f"Averaging window: {config.inference.averaging_window_ms}ms")
        print(f"Suppression: {config.inference.suppression_ms}ms\n")
        
        try:
            while True:
                try:
                    chunk = audio_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                
                # Update rolling buffer
                buffer = np.roll(buffer, -len(chunk))
                buffer[-len(chunk):] = chunk
                
                # Normalize
                normalized = buffer / (np.linalg.norm(buffer) + 1e-6)
                
                # Detect
                if use_torchscript:
                    # Convert to features and run
                    mfcc = detector.preprocessor.extract_mfcc(normalized, add_noise=False)
                    mfcc = detector.preprocessor.pad_or_truncate(mfcc, target_length=98)
                    
                    if config.model.use_2d_conv:
                        features = torch.FloatTensor(mfcc.T).unsqueeze(0).unsqueeze(0)
                    else:
                        features = torch.FloatTensor(mfcc[:, 0]).unsqueeze(0).unsqueeze(0)
                    
                    probs = runner(features)
                    hello_idx = config.labels.index("hello")
                    hello_prob = float(probs[0, hello_idx])
                    
                    # Use sliding window logic
                    now = time.time()
                    detector.detection_scores.append((now, hello_prob))
                    cutoff = now - config.inference.averaging_window_ms / 1000
                    while detector.detection_scores and detector.detection_scores[0][0] < cutoff:
                        detector.detection_scores.popleft()
                    
                    avg_score = sum(s for _, s in detector.detection_scores) / max(len(detector.detection_scores), 1)
                    
                    time_since_last = now - detector.last_detection_time
                    is_detected = (
                        avg_score >= config.inference.detection_threshold and
                        time_since_last >= config.inference.suppression_ms / 1000
                    )
                    
                    if is_detected:
                        print(f"[{time.strftime('%H:%M:%S')}] 🎤 HELLO DETECTED! (score={avg_score:.3f})")
                        detector.last_detection_time = now
                        detector.detection_scores.clear()
                        time.sleep(config.inference.suppression_ms / 1000)
                else:
                    is_detected, avg_score = detector.predict(normalized)
                    if is_detected:
                        print(f"[{time.strftime('%H:%M:%S')}] 🎤 HELLO DETECTED! (score={avg_score:.3f})")
                        time.sleep(config.inference.suppression_ms / 1000)
                
        except KeyboardInterrupt:
            print("\nStopping stream.")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced real-time hello detector")
    parser.add_argument("--model", type=str, default="models/hello_model.pth",
                      help="Path to model")
    parser.add_argument("--threshold", type=float, default=None,
                      help="Detection threshold (overrides config)")
    parser.add_argument("--window-ms", type=int, default=None,
                      help="Sliding window size in ms")
    
    args = parser.parse_args()
    
    config = ProjectConfig()
    if args.threshold:
        config.inference.detection_threshold = args.threshold
    if args.window_ms:
        config.inference.window_ms = args.window_ms
    
    stream_detection(args.model, config)

