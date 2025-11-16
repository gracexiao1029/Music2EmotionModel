"""
Audio format conversion and preprocessing script
Convert MP3 files from MEMD_audio directory to WAV format and organize them in audio_wav directory
"""
import os
import librosa
import soundfile as sf
from tqdm import tqdm
import numpy as np

# Configuration
INPUT_DIR = "MEMD_audio"
OUTPUT_DIR = "audio_wav"
SR = 22050  # Sample rate, consistent with training script
TARGET_FORMAT = "wav"

def convert_mp3_to_wav(input_file, output_file, sr=SR):
    """Convert MP3 file to WAV format"""
    try:
        # Load audio using librosa (automatically handles format conversion)
        y, original_sr = librosa.load(input_file, sr=sr, mono=True)
        
        # Save as WAV format
        sf.write(output_file, y, sr)
        return True
    except Exception as e:
        print(f"Error converting {input_file}: {e}")
        return False

def get_song_id(filename):
    """Extract song_id from filename"""
    # Example: "10.mp3" -> 10, "1000.mp3" -> 1000
    base = os.path.splitext(filename)[0]
    try:
        return int(base)
    except ValueError:
        return None

def main():
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Get all audio files
    audio_files = []
    if os.path.exists(INPUT_DIR):
        for filename in os.listdir(INPUT_DIR):
            if filename.lower().endswith(('.mp3', '.wav', '.m4a', '.flac', '.ogg')):
                audio_files.append(filename)
    
    if not audio_files:
        print(f"No audio files found in {INPUT_DIR} directory")
        return
    
    print(f"Found {len(audio_files)} audio files")
    print(f"Starting conversion to {OUTPUT_DIR} directory...")
    
    # Convert files
    success_count = 0
    failed_count = 0
    skipped_count = 0
    
    for filename in tqdm(audio_files, desc="Converting audio"):
        input_path = os.path.join(INPUT_DIR, filename)
        song_id = get_song_id(filename)
        
        if song_id is None:
            print(f"Warning: Cannot extract song_id from filename: {filename}")
            failed_count += 1
            continue
        
        # Output filename format: {song_id}.wav
        output_filename = f"{song_id}.wav"
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        
        # Skip if file already exists
        if os.path.exists(output_path):
            skipped_count += 1
            continue
        
        # Convert file
        if convert_mp3_to_wav(input_path, output_path, sr=SR):
            success_count += 1
        else:
            failed_count += 1
    
    print("\nConversion completed!")
    print(f"Success: {success_count}")
    print(f"Failed: {failed_count}")
    print(f"Skipped (already exists): {skipped_count}")
    print(f"Total: {len(audio_files)}")
    
    # Check files in output directory
    if os.path.exists(OUTPUT_DIR):
        wav_files = [f for f in os.listdir(OUTPUT_DIR) if f.endswith('.wav')]
        print(f"\n{OUTPUT_DIR} directory contains {len(wav_files)} WAV files")
        
        # Display some example files
        if wav_files:
            print("\nExample files:")
            for f in sorted(wav_files)[:10]:
                filepath = os.path.join(OUTPUT_DIR, f)
                try:
                    y, sr = librosa.load(filepath, sr=None, duration=0.1)
                    duration = librosa.get_duration(y=y, sr=sr)
                    print(f"  {f}: {duration:.2f}s, {sr}Hz")
                except:
                    print(f"  {f}: (cannot read)")

if __name__ == "__main__":
    main()

