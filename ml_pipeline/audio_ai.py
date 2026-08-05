import whisper
import librosa
import numpy as np
import noisereduce as nr
import soundfile as sf
import subprocess
import tempfile
import os

# Expanded filler word list (Standard English + Indian English variants)
FILLER_WORDS = [
    # Standard English fillers
    " um ", " uh ", " like ", " you know ", " basically ", " literally ",
    " actually ", " so ", " right ", " i mean ", " kind of ", " sort of ",
    " anyway ", " okay so ", " well ", " hmm ",
    # Indian English fillers
    " aa ", " aaa ", " ah ", " aah ", " matlab ", " haan ", " na ",
    " arre ", " toh ", " woh ", " yaar ", " bas ", " accha ", " theek hai "
]

# Single-syllable words that are almost always fillers when stretched > 0.8s
PROLONGED_FILLER_SOUNDS = {
    "aa", "aaa", "ah", "aah", "um", "uh", "hmm", "hm",
    "haan", "na", "toh", "woh", "so", "well", "and"
}

def extract_audio_from_video(video_path):
    """Extract audio from video and save as a temporary WAV file using ffmpeg."""
    temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
    temp_audio.close()
    cmd = [
        'ffmpeg', '-y', '-i', video_path,
        '-ac', '1', '-ar', '16000',
        '-vn', temp_audio.name
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return temp_audio.name


def clean_audio(audio_path):
    """
    Remove background noise from audio using noisereduce.
    Reads the raw WAV, applies spectral noise gating, and saves
    a cleaned version back to the same path.
    """
    print("Cleaning audio (noise removal)...")
    try:
        y, sr = librosa.load(audio_path, sr=16000)
        # noisereduce uses the first 0.5s of audio as the noise profile
        # (assumes the start of the recording has background noise without speech)
        noise_sample = y[:int(sr * 0.5)]
        y_clean = nr.reduce_noise(y=y, sr=sr, y_noise=noise_sample, prop_decrease=0.75)
        sf.write(audio_path, y_clean, sr)
    except Exception as e:
        print(f"Noise reduction skipped: {e}")


def analyze_prosody(audio_path):
    """Analyze pitch, energy, and speaking rate using librosa."""
    y, sr = librosa.load(audio_path, sr=16000)

    # --- Pitch (F0) using pyin ---
    f0, voiced_flag, voiced_probs = librosa.pyin(
        y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7')
    )
    valid_f0 = f0[~np.isnan(f0)]
    mean_pitch = round(float(np.mean(valid_f0)), 1) if len(valid_f0) > 0 else 0.0
    pitch_variation = round(float(np.std(valid_f0)), 1) if len(valid_f0) > 0 else 0.0

    # --- Energy (RMS) ---
    rms = librosa.feature.rms(y=y)[0]
    mean_energy = round(float(np.mean(rms)), 4)
    energy_variation = round(float(np.std(rms)), 4)

    return {
        "mean_pitch_hz": mean_pitch,
        "pitch_variation_hz": pitch_variation,
        "mean_energy": mean_energy,
        "energy_variation": energy_variation
    }


def detect_pauses(whisper_segments, min_pause_sec=0.5):
    """Detect pauses between Whisper transcript segments."""
    pauses = []
    for i in range(1, len(whisper_segments)):
        gap = whisper_segments[i]["start"] - whisper_segments[i - 1]["end"]
        if gap >= min_pause_sec:
            pauses.append(round(gap, 2))
    
    num_pauses = len(pauses)
    avg_pause = round(float(np.mean(pauses)), 2) if pauses else 0.0
    total_pause_time = round(float(np.sum(pauses)), 2) if pauses else 0.0

    return {
        "num_pauses": num_pauses,
        "avg_pause_sec": avg_pause,
        "total_pause_sec": total_pause_time
    }


import string

def count_fillers(text):
    """Count filler words in transcript using text matching, ignoring punctuation."""
    # Remove punctuation so words like "uh," or "um." can match our space-padded list " uh "
    translator = str.maketrans('', '', string.punctuation)
    text_clean = text.translate(translator)
    
    text_lower = " " + text_clean.lower() + " "
    filler_details = {}
    total = 0
    for filler in FILLER_WORDS:
        count = text_lower.count(filler)
        if count > 0:
            filler_details[filler.strip()] = count
            total += count
    return total, filler_details


def detect_prolonged_fillers(words):
    """
    Detect filler sounds using Whisper's word-level timestamps.
    A single-syllable word (e.g., 'aa', 'uh') that takes longer than
    0.8 seconds to say is almost certainly a prolonged filler sound.
    This catches sounds that text matching alone would miss.
    """
    prolonged_count = 0
    prolonged_details = {}

    for word_info in words:
        word = word_info.get("word", "").strip().lower()
        start = word_info.get("start", 0)
        end = word_info.get("end", 0)
        duration = end - start

        # Check if this word is a known single-syllable filler sound
        # AND if it was stretched beyond 0.8 seconds (a clear filler indicator)
        if word in PROLONGED_FILLER_SOUNDS and duration >= 0.8:
            prolonged_count += 1
            prolonged_details[word] = prolonged_details.get(word, 0) + 1

    return prolonged_count, prolonged_details


def analyze_audio(video_path):
    print("Extracting audio from video...")
    audio_path = extract_audio_from_video(video_path)

    try:
        # --- Step 1: Noise Removal ---
        clean_audio(audio_path)

        # --- Step 2: Whisper Transcription ---
        # Using 'small' model for significantly better accuracy than 'base'
        # with minimal extra RAM usage on a 16GB machine.
        print("Loading Whisper model...")
        model = whisper.load_model("small")
        print("Transcribing audio...")
        # initial_prompt primes Whisper to keep filler words instead of silently deleting them.
        # Without this, Whisper aggressively cleans up "umm", "uh", "aa" etc from the transcript.
        FILLER_PROMPT = (
    "Umm,um, let me think, uh, so basically, like, you know, aa, ah, er, erm, "
    "I mean, right, hmm, mhm, uh-huh, actually, well, kind of, sorta, yeah, "
    "okay, alright, anyway, anyways, I guess, you see. Haan, matlab, toh, "
    "arre, yaar, bhai, theek hai, achha, haina, waise, dekho, phir, "
    "kya bolte hain, jaise."
)
        result = model.transcribe(
            audio_path,
            word_timestamps=True,
            initial_prompt=FILLER_PROMPT,
            language="en"  # Force English — prevents wrong language detection
        )
        text = result["text"]
        segments = result.get("segments", [])

        # --- Extract word-level timestamp data from all segments ---
        all_words = []
        for seg in segments:
            words_in_seg = seg.get("words", [])
            all_words.extend(words_in_seg)

        # --- Speaking Rate (WPM) ---
        word_count = len(text.split())
        duration = segments[-1]["end"] if segments else 1
        wpm = round((word_count / duration) * 60)

        # --- Filler Words (Method 1: Text Matching) ---
        text_fillers, filler_breakdown = count_fillers(text)

        # --- Filler Words (Method 2: Prolonged Sound Detection via timestamps) ---
        prolonged_fillers, prolonged_breakdown = detect_prolonged_fillers(all_words)

        # --- Combine both methods, avoiding double-counting ---
        # We merge the breakdowns and take the max count per word to avoid duplicates
        combined_breakdown = dict(filler_breakdown)
        for word, count in prolonged_breakdown.items():
            # Only add if prolonged method found MORE instances than text matching
            existing = combined_breakdown.get(word, 0)
            if count > existing:
                combined_breakdown[word] = count
                prolonged_fillers -= existing  # remove already-counted ones

        total_fillers = sum(combined_breakdown.values())
        fillers_per_minute = round(total_fillers / (duration / 60), 2) if duration > 0 else 0

        # --- Pauses ---
        pause_data = detect_pauses(segments)

        # --- Prosody (Pitch + Energy) ---
        print("Analyzing prosody (pitch, energy)...")
        prosody_data = analyze_prosody(audio_path)

    finally:
        # Clean up temp audio file
        if os.path.exists(audio_path):
            os.remove(audio_path)

    return {
        "transcript": text,
        "word_count": word_count,
        "duration_sec": round(duration, 1),
        # Speaking Rate
        "wpm": wpm,
        # Filler Words (combined from both detection methods)
        "fillers": total_fillers,
        "fillers_per_minute": fillers_per_minute,
        "filler_breakdown": combined_breakdown,
        # Pauses
        "num_pauses": pause_data["num_pauses"],
        "avg_pause_sec": pause_data["avg_pause_sec"],
        "total_pause_sec": pause_data["total_pause_sec"],
        # Prosody
        "mean_pitch_hz": prosody_data["mean_pitch_hz"],
        "pitch_variation_hz": prosody_data["pitch_variation_hz"],
        "mean_energy": prosody_data["mean_energy"],
        "energy_variation": prosody_data["energy_variation"]
    }