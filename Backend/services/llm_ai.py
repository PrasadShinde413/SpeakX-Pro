import requests

def generate_feedback(audio_results, video_results, nlp_results):
    """
    Passes all collected metrics to the local Ollama LLM to generate a personalized coaching report.
    """
    
    # Safely extract values in case some metrics failed
#     wpm = audio_results.get("wpm", 0)
#     fillers = audio_results.get("fillers", 0)
#     transcript = audio_results.get("transcript", "")
    
#     eye_contact = video_results.get("eye_contact_pct", 0)
#     posture = video_results.get("posture_dominant", "Unknown")
#     smile_pct = video_results.get("smile_pct", 0)
#     gestures = video_results.get("gesture_active_pct", 0)
    
#     grammar = nlp_results.get("grammar_errors", 0)
#     readability = nlp_results.get("readability", "Unknown")
#     ttr = nlp_results.get("vocabulary_ttr", 0)
    
#     prompt = f"""
# You are an expert, encouraging public speaking and confidence coach. Your student just recorded their daily 5-minute speaking practice video. 
# Analyze their exact metrics below and write a highly personalized, constructive feedback report.

# === STUDENT'S METRICS ===
# [AUDIO & SPEECH]
# - Speaking Pace: {wpm} Words Per Minute (Ideal is 130-160 WPM)
# - Filler Words Used: {fillers}
# - Transcript: "{transcript}"

# [VIDEO & BODY LANGUAGE]
# - Eye Contact / Camera Focus: {eye_contact}% (Higher is better)
# - Dominant Posture: {posture}
# - Smiling Frequency: {smile_pct}%
# - Hand Gesture Activity: {gestures}%

# [LANGUAGE & NLP]
# - Grammar Errors: {grammar}
# - Readability Level: {readability}
# - Vocabulary Richness: {ttr} (Scale of 0.0 to 1.0, higher is better)

# === INSTRUCTIONS ===
# Write a short, engaging feedback report. Do not just list the metrics back to the student. Interpret what the metrics *mean* for their confidence and communication style. 
# Your response MUST be formatted with exactly these three bold headers:

# **🌟 The Highlight**
# (Write 2 sentences praising them for their strongest metric today. Explain why this specific strength makes them a better communicator.)

# **📈 The Polish**
# (Write 2 sentences gently pointing out their weakest metric today. Explain how this specific habit might negatively impact their audience's perception of them.)

# **🎯 The Challenge**
# (Write 1 highly specific, actionable goal for them to focus on in tomorrow's 5-minute video to fix the weakness mentioned above.)

# Keep your tone warm, professional, and highly encouraging!
# """
# Audio variables
    wpm = audio_results.get("wpm", 0)
    fillers = audio_results.get("fillers", 0)
    fillers_per_min = audio_results.get("fillers_per_min", 0.0)
    total_pauses = audio_results.get("total_pauses", 0)
    avg_pause = audio_results.get("avg_pause", 0.0)
    total_pause_time = audio_results.get("total_pause_time", 0.0)
    mean_pitch = audio_results.get("mean_pitch", 0.0)
    pitch_variation = audio_results.get("pitch_variation", 0.0)
    transcript = audio_results.get("transcript", "")
    
    # Video variables
    eye_contact = video_results.get("eye_contact_pct", 0)
    dominant_head = video_results.get("head_pose_dominant", "Unknown")
    head_pose_forward = video_results.get("head_pose_forward_pct", 0)
    dominant_posture = video_results.get("posture_dominant", "Unknown")
    upright_posture = video_results.get("posture_upright_pct", 0)
    smile_pct = video_results.get("smile_pct", 0)
    gestures = video_results.get("gesture_active_pct", 0)
    
    # NLP variables
    readability = nlp_results.get("readability", "Unknown")
    grade_level = nlp_results.get("grade_level", "N/A")
    reading_ease = nlp_results.get("reading_ease", 0.0)
    ttr = nlp_results.get("vocabulary_ttr", 0.0)
    coherence_score = nlp_results.get("coherence_score", 0.0)
    sentences = nlp_results.get("sentences", 0)
    avg_sentence_length = nlp_results.get("avg_sentence_length", 0.0)
    grammar = nlp_results.get("grammar_errors", "N/A")

    prompt = f"""
You are an elite executive communication and public speaking coach. Your student is completing a 75-day challenge to achieve peak professional communication skills for high-stakes corporate environments.

Analyze their daily 5-minute video metrics and transcript to write a highly personalized, detailed, and constructive feedback report. Do not just regurgitate the numbers; interpret how these metrics combine to affect their perceived confidence, authority, and engagement.

=== STUDENT'S RAW METRICS ===
[AUDIO & PROSODY]
- Speaking Pace: {wpm} WPM (Target: 130-160)
- Filler Words: {fillers} (Rate: {fillers_per_min} per min)
- Pauses: {total_pauses} pauses, {avg_pause}s avg duration, {total_pause_time}s total
- Pitch: {mean_pitch} Hz Mean, {pitch_variation} Hz Variation

[VIDEO & BODY LANGUAGE]
- Eye Contact: {eye_contact}% 
- Head Pose: {dominant_head} ({head_pose_forward}% forward)
- Posture: {dominant_posture} ({upright_posture}% upright)
- Hand Gestures: {gestures}%
- Smiling Frequency: {smile_pct}%

[NLP & CONTENT]
- Readability: {readability} (Grade {grade_level}, Ease: {reading_ease})
- Vocab Richness (TTR): {ttr}
- Coherence Score: {coherence_score}
- Sentences: {sentences} (Avg Length: {avg_sentence_length} words)
- Grammar Errors: {grammar}
- Transcript: "{transcript}"

=== OUTPUT FORMAT ===
Provide your analysis using exactly the following Markdown structure. Keep the tone professional, highly encouraging, and analytically precise.

### 💯 Performance Scores
Evaluate the performance on a strict scale of 0.0 to 10.0 based on the raw data.
* **Confidence:** X.X/10 (Assess using eye contact, upright posture, head pose, and pitch variation)
* **Fluency:** X.X/10 (Assess using WPM, filler word count, and pause frequency/durations)
* **English Proficiency:** X.X/10 (Assess using grammar errors, sentence structures, and readability ease)
* **Communication Impact:** X.X/10 (Assess using coherence score, vocabulary richness/TTR, and gesture activity)
* **Vocal Engagement:** X.X/10 (Assess using pitch variations, mean pitch stability, and natural pause pacing)
* **Physical Presence:** X.X/10 (Assess using the combination of smiling frequency, posture, and forward head placement)

### 📊 Executive Assessment
(Provide a 2-3 sentence summary of their overall communication style today. Cross-reference their metrics—for example, how their pace and pause durations affected their coherence, or how their posture and eye contact projected authority.)

### 🌟 Core Strengths
* **[Insert Strength 1 - e.g., Vocal Dynamics]:** (Explain what they did well using specific metrics and why it positively impacts the listener.)
* **[Insert Strength 2 - e.g., Body Language]:** (Explain the second strength, tying it to a specific metric.)

### 📈 Areas for Polish
* **[Insert Weakness 1 - e.g., Pacing & Pauses]:** (Identify a weakness based on the data. Explain how it might make them appear nervous, unpolished, or disconnected.)
* **[Insert Weakness 2 - e.g., Content Clarity]:** (Identify a second area for improvement using the NLP/Audio metrics.)

### 🎯 Tomorrow's Action Plan
Provide exactly TWO highly specific, actionable physical or vocal exercises to practice in tomorrow's video to directly correct the weaknesses mentioned above. (e.g., "Take a deliberate 2-second pause after every main idea to reduce your filler rate of {fillers_per_min}/min.")
"""
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": "qwen2.5", # Ensure model name is lowercase for Ollama
        "prompt": prompt,
        "stream": False
    }
    
    try:
        # Added a timeout so the Streamlit UI doesn't freeze forever if Ollama is off
        response = requests.post(url, json=payload, timeout=180)
        
        if response.status_code == 200:
            return response.json().get("response", "Error parsing LLM response.")
        else:
            return f"❌ Ollama Error ({response.status_code}): {response.text}"
            
    except requests.exceptions.ConnectionError:
        return "❌ Connection Error: Could not connect to local LLM. Please make sure the Ollama app is running on your machine."
    except requests.exceptions.Timeout:
        return "❌ Timeout Error: The LLM took too long to respond. The model might be too heavy for your machine's RAM."
    except Exception as e:
        return f"❌ Unexpected Error: {str(e)}"