"""
Utility script to help create sample training data
Records audio samples for keyword spotting training
"""

import numpy as np
import sounddevice as sd
import scipy.io.wavfile as wavfile
import os
import argparse
from pathlib import Path


def record_sample(output_path, duration_seconds=1.5, sample_rate=16000):
    """
    Record audio sample and save as WAV file
    
    Args:
        output_path: Path to save WAV file
        duration_seconds: Recording duration
        sample_rate: Audio sample rate
    """
    print(f"\nRecording for {duration_seconds} seconds...")
    print("Speak now!")
    
    try:
        audio = sd.rec(
            int(duration_seconds * sample_rate),
            samplerate=sample_rate,
            channels=1,
            dtype='float32'
        )
        sd.wait()  # Wait until recording is finished
        
        # Convert to int16 for WAV file
        audio_int16 = (audio.flatten() * 32767).astype(np.int16)
        
        # Save WAV file
        wavfile.write(output_path, sample_rate, audio_int16)
        print(f"Saved to: {output_path}")
        
        return True
        
    except Exception as e:
        print(f"Error recording: {e}")
        return False


def interactive_recording(class_name, output_dir, num_samples=10):
    """
    Interactive recording session for collecting training data
    
    Args:
        class_name: Class name ("hello" or "other")
        output_dir: Directory to save recordings
        num_samples: Number of samples to record
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Recording {num_samples} samples for class: '{class_name}'")
    print(f"Output directory: {output_dir}")
    print(f"{'='*60}")
    
    for i in range(num_samples):
        filename = f"{class_name}_{i+1:03d}.wav"
        output_path = os.path.join(output_dir, filename)
        
        print(f"\nSample {i+1}/{num_samples}")
        record_sample(output_path, duration_seconds=1.5)
        
        if i < num_samples - 1:
            response = input("\nPress Enter for next sample, or 'q' to quit: ")
            if response.lower() == 'q':
                break
    
    print(f"\nCompleted! Recorded samples saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Record audio samples for keyword spotting training"
    )
    parser.add_argument("--class", type=str, required=True,
                       choices=["hello", "other"],
                       help="Class name (hello or other)")
    parser.add_argument("--output_dir", type=str,
                       default=None,
                       help="Output directory (default: data/{class})")
    parser.add_argument("--num_samples", type=int, default=10,
                       help="Number of samples to record")
    parser.add_argument("--single", type=str, default=None,
                       help="Record single file (output path)")
    
    args = parser.parse_args()
    
    # Determine output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = os.path.join("data", args.__dict__["class"])
    
    if args.single:
        # Record single file
        record_sample(args.single, duration_seconds=1.5)
    else:
        # Interactive recording session
        interactive_recording(args.__dict__["class"], output_dir, args.num_samples)


if __name__ == "__main__":
    try:
        import sounddevice
        import scipy
    except ImportError:
        print("ERROR: Required packages not installed.")
        print("Install with: pip install sounddevice scipy")
        exit(1)
    
    main()

