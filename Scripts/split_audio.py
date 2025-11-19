"""
Audio segmentation script
Split audio files in audio_wav directory into fixed-length segments
"""
import os
import librosa
import soundfile as sf
from tqdm import tqdm
import numpy as np

# Configuration
INPUT_DIR = "audio_wav"
OUTPUT_DIR = "audio_segments"  # Output directory for segments
SEGMENT_DURATION = 30  # Length of each segment (seconds)
OVERLAP = 0  # Overlap between segments (seconds), 0 means no overlap
SR = 22050  # Sample rate

def split_audio_file(input_file, output_dir, segment_duration=30, overlap=0, sr=22050):
    """Split audio file into multiple segments"""
    try:
        # Load audio
        y, original_sr = librosa.load(input_file, sr=sr, mono=True)
        duration = len(y) / sr
        
        if duration < segment_duration:
            # If audio is too short, save only one segment (pad or truncate)
            if len(y) < segment_duration * sr:
                y = np.pad(y, (0, int(segment_duration * sr) - len(y)))
            output_file = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(input_file))[0]}_seg0.wav")
            sf.write(output_file, y[:int(segment_duration * sr)], sr)
            return 1
        
        # Calculate number of segments and step size
        step = int((segment_duration - overlap) * sr)
        num_segments = int(np.ceil((len(y) - segment_duration * sr) / step)) + 1
        
        saved_count = 0
        base_name = os.path.splitext(os.path.basename(input_file))[0]
        
        for i in range(num_segments):
            start = i * step
            end = start + int(segment_duration * sr)
            
            if end > len(y):
                # Last segment, pad if not long enough
                segment = y[start:]
                if len(segment) < segment_duration * sr:
                    segment = np.pad(segment, (0, int(segment_duration * sr) - len(segment)))
                else:
                    segment = segment[:int(segment_duration * sr)]
            else:
                segment = y[start:end]
            
            output_file = os.path.join(output_dir, f"{base_name}_seg{i:03d}.wav")
            sf.write(output_file, segment, sr)
            saved_count += 1
        
        return saved_count
    except Exception as e:
        print(f"Error processing {input_file}: {e}")
        return 0

def main():
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Get all WAV files
    wav_files = [f for f in os.listdir(INPUT_DIR) if f.endswith('.wav')]
    
    if not wav_files:
        print(f"No WAV files found in {INPUT_DIR} directory")
        return
    
    print(f"Found {len(wav_files)} WAV files")
    print(f"Segmentation config: {SEGMENT_DURATION}s per segment, {OVERLAP}s overlap")
    print(f"Starting segmentation to {OUTPUT_DIR} directory...")
    
    total_segments = 0
    success_count = 0
    failed_count = 0
    
    for filename in tqdm(wav_files, desc="Splitting audio"):
        input_path = os.path.join(INPUT_DIR, filename)
        segments = split_audio_file(input_path, OUTPUT_DIR, SEGMENT_DURATION, OVERLAP, SR)
        
        if segments > 0:
            success_count += 1
            total_segments += segments
        else:
            failed_count += 1
    
    print("\nSegmentation completed!")
    print(f"Successfully processed: {success_count} files")
    print(f"Failed: {failed_count} files")
    print(f"Total generated: {total_segments} audio segments")
    
    # Statistics for output directory
    if os.path.exists(OUTPUT_DIR):
        segment_files = [f for f in os.listdir(OUTPUT_DIR) if f.endswith('.wav')]
        print(f"\n{OUTPUT_DIR} directory contains {len(segment_files)} segment files")
        
        # Display some examples
        if segment_files:
            print("\nExample segments:")
            for f in sorted(segment_files)[:5]:
                filepath = os.path.join(OUTPUT_DIR, f)
                try:
                    y, sr = librosa.load(filepath, sr=None, duration=0.1)
                    duration = librosa.get_duration(y=y, sr=sr)
                    print(f"  {f}: {duration:.2f}s")
                except:
                    print(f"  {f}: (cannot read)")

if __name__ == "__main__":
    main()

