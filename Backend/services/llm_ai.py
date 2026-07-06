import requests


def generate_feedback(audio_results, video_results, nlp_results):
    """
    Generate AI coaching report using Ollama.
    """

    # ==========================
    # AUDIO VARIABLES
    # ==========================
    wpm = audio_results.get("wpm", 0)
    fillers = audio_results.get("fillers", 0)
    fillers_per_min = audio_results.get("fillers_per_min", 0.0)

    total_pauses = audio_results.get("total_pauses", 0)
    avg_pause = audio_results.get("avg_pause", 0.0)
    total_pause_time = audio_results.get("total_pause_time", 0.0)

    mean_pitch = audio_results.get("mean_pitch", 0.0)
    pitch_variation = audio_results.get("pitch_variation", 0.0)

    transcript = audio_results.get("transcript", "")

    # ==========================
    # VIDEO VARIABLES
    # ==========================
    eye_contact = video_results.get("eye_contact_pct", 0)

    dominant_head = video_results.get(
        "head_pose_dominant",
        "Unknown"
    )

    head_pose_forward = video_results.get(
        "head_pose_forward_pct",
        0
    )

    dominant_posture = video_results.get(
        "posture_dominant",
        "Unknown"
    )

    upright_posture = video_results.get(
        "posture_upright_pct",
        0
    )

    smile_pct = video_results.get(
        "smile_pct",
        0
    )

    gestures = video_results.get(
        "gesture_active_pct",
        0
    )

    # ==========================
    # NLP VARIABLES
    # ==========================
    readability = nlp_results.get(
        "readability",
        "Unknown"
    )

    grade_level = nlp_results.get(
        "grade_level",
        "N/A"
    )

    reading_ease = nlp_results.get(
        "reading_ease",
        0.0
    )

    ttr = nlp_results.get(
        "vocabulary_ttr",
        0.0
    )

    coherence_score = nlp_results.get(
        "coherence_score",
        0.0
    )

    sentences = nlp_results.get(
        "sentences",
        0
    )

    avg_sentence_length = nlp_results.get(
        "avg_sentence_length",
        0.0
    )

    # ==========================
    # PROMPT
    # ==========================
    prompt = f"""
You are an expert English Communication Coach, Public Speaking Trainer, and Grammar Evaluator.

Your task is to evaluate the student's spoken English performance using:

1. Audio metrics
2. Video metrics
3. NLP metrics
4. Transcript

IMPORTANT RULES:

- Evaluate ALL areas equally.
- Do NOT over-focus on pauses.
- Use transcript to detect grammar mistakes.
- Use transcript to evaluate vocabulary and communication quality.
- Ignore normal speech hesitations like "um", "uh", and self-corrections unless they create genuine grammar errors.
- If grammar mistakes exist, provide corrected versions.
- Areas for Polish MUST identify the TWO weakest dimensions.
- The two weaknesses should come from different categories whenever possible.
- Tomorrow's Action Plan MUST directly address the weaknesses identified.
- Do not simply repeat the metrics.
- Explain why the metrics matter.

Evaluation Weight:

Audio = 30%
Video = 25%
Grammar & English = 25%
Communication & Content = 20%

====================================================
AUDIO METRICS
====================================================

Speaking Pace: {wpm} WPM

Filler Words: {fillers}

Fillers Per Minute: {fillers_per_min}

Total Pauses: {total_pauses}

Average Pause Duration: {avg_pause}

Total Pause Time: {total_pause_time}

Mean Pitch: {mean_pitch}

Pitch Variation: {pitch_variation}

====================================================
VIDEO METRICS
====================================================

Eye Contact: {eye_contact}%

Head Pose Dominant: {dominant_head}

Forward Head Percentage: {head_pose_forward}%

Posture Dominant: {dominant_posture}

Upright Posture Percentage: {upright_posture}%

Hand Gesture Activity: {gestures}%

Smile Frequency: {smile_pct}%

====================================================
NLP METRICS
====================================================

Readability: {readability}

Grade Level: {grade_level}

Reading Ease: {reading_ease}

Vocabulary Richness (TTR): {ttr}

Coherence Score: {coherence_score}

Total Sentences: {sentences}

Average Sentence Length: {avg_sentence_length}

====================================================
TRANSCRIPT
====================================================

{transcript}

====================================================
OUTPUT FORMAT
====================================================

# 💯 Performance Scores

Confidence: X.X/10

Fluency: X.X/10

English Proficiency: X.X/10

Communication Impact: X.X/10

Vocal Engagement: X.X/10

Physical Presence: X.X/10

Overall Performance: X.X/10

Performance Level:
(Beginner / Elementary / Intermediate / Upper Intermediate / Advanced / Interview Ready)

----------------------------------------------------

# 📊 Executive Assessment

Provide 2-3 paragraphs.

Discuss:

- Confidence
- Fluency
- Communication
- Body Language
- Grammar
- Speaking Style

Interpret metrics instead of repeating them.

----------------------------------------------------

# ✅ Grammar Review

Grammar Errors Found: X

For each grammar mistake provide:

❌ Original

✅ Correct

📚 Rule Violated

💡 Explanation

Maximum 5 mistakes.

If no mistakes exist, explicitly say:

"No significant grammar mistakes detected."

----------------------------------------------------

# 🌟 Core Strengths

Provide exactly TWO strengths.

For each:

- Metric evidence
- Why it helps communication
- Impact on listener

----------------------------------------------------

# 📈 Areas for Polish

Provide exactly TWO weaknesses.

Choose the weakest dimensions.

Do NOT choose pauses repeatedly unless pause metrics are genuinely poor.

Explain:

- Why it is a weakness
- Evidence from metrics
- Impact on audience

----------------------------------------------------

# 🎯 Tomorrow's Action Plan

Provide EXACTLY TWO exercises.

For each exercise:

Exercise Name:

Duration:

Steps:

Target Metric:

Expected Improvement:

----------------------------------------------------

# ⭐ Final Coach's Advice

Provide one motivating paragraph.

Mention the single most important improvement area for tomorrow.
"""

    # ==========================
    # OLLAMA REQUEST
    # ==========================
    url = "http://localhost:11434/api/generate"

    payload = {
        # "model": "qwen2.5",
        # "model": "Qwen2.5:14b-instruct ",
        "model": "Qwen2.5:7b-instruct", 
        # "model": "Qwen2.5:latest",
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.4,
            # "top_p": 0.9
        }
    }

    try:
        response = requests.post(url,json=payload,timeout=2000)

        if response.status_code == 200:
            result = response.json()
            return result.get(
                "response",
                "Error parsing LLM response."
            )

        return f"❌ Ollama Error ({response.status_code}): {response.text}"

    except requests.exceptions.ConnectionError:
        return (
            "❌ Could not connect to Ollama.\n"
            "Make sure Ollama is running."
        )

    except requests.exceptions.Timeout:
        return (
            "❌ Model timeout.\n"
            "Try using a smaller model or increasing timeout."
        )

    except Exception as e:
        return f"❌ Unexpected Error: {str(e)}"