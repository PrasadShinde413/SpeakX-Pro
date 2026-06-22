import whisper
import librosa
import numpy as np
import subprocess
import tempfile
import os

# Expanded filler word list
FILLER_WORDS = [
    " um ", " uh ", " like ", " you know ", " basically ", " literally ",
    " actually ", " so ", " right ", " i mean ", " kind of ", " sort of ",
    " anyway ", " okay so ", " well ", " hmm "
]

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


def count_fillers(text):
    """Count filler words in transcript."""
    text_lower = " " + text.lower() + " "
    filler_details = {}
    total = 0
    for filler in FILLER_WORDS:
        count = text_lower.count(filler)
        if count > 0:
            filler_details[filler.strip()] = count
            total += count
    return total, filler_details


def analyze_audio(video_path):
    print("Extracting audio from video...")
    audio_path = extract_audio_from_video(video_path)

    try:
        # --- Whisper Transcription ---
        print("Loading Whisper model...")
        model = whisper.load_model("base")
        print("Transcribing audio...")
        result = model.transcribe(audio_path)
        text = result["text"]
        segments = result.get("segments", [])

        # --- Speaking Rate (WPM) ---
        word_count = len(text.split())
        duration = segments[-1]["end"] if segments else 1
        wpm = round((word_count / duration) * 60)

        # --- Filler Words ---
        total_fillers, filler_breakdown = count_fillers(text)
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
        # Filler Words
        "fillers": total_fillers,
        "fillers_per_minute": fillers_per_minute,
        "filler_breakdown": filler_breakdown,
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