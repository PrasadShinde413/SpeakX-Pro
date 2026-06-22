# 🎥 Confidence Coach AI

**Confidence Coach** is an intelligent, privacy-first AI platform designed to help individuals dramatically improve their public speaking skills, communication fluency, and self-confidence through the automated analysis of daily 5-minute video recordings.

---

## 🎯 Aim & Objectives
The goal of this project is to provide a fully automated, multi-modal feedback loop for speakers without requiring manual human review. 

*   **Vocal Analysis:** Uses OpenAI Whisper and Librosa to measure speaking pace, pause durations, vocal pitch, and the frequency of filler words.
*   **Body Language Tracking:** Uses Google MediaPipe vision models to evaluate non-verbal cues, including eye contact, posture alignment, hand gestures, and facial expressions.
*   **Linguistic Evaluation:** Uses NLP algorithms (NLTK, Textstat, LanguageTool) to assess grammar correctness, vocabulary richness, and sentence coherence.
*   **AI Coaching Synthesis:** Feeds the extracted metrics into a locally hosted Large Language Model (e.g., Qwen/Llama via Ollama) to generate personalized, highly actionable coaching feedback.

---

## 🚀 How to Run Locally

### 1. Prerequisites
- Python 3.10+
- [Ollama](https://ollama.com/) installed and running locally.
- A supported LLM pulled in Ollama (e.g., `ollama run qwen2.5`)

### 2. Setup
Clone the repository and install the dependencies:
```bash
git clone <your-repo-url>
cd "Video Analyzer/Backend"

# Create and activate virtual environment
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
# source venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

### 3. Run the App
Launch the Streamlit dashboard:
```bash
python -m streamlit run app.py
```
*Note: Make sure your Ollama application is running in the background before analyzing a video so the Coach Feedback feature works!*

---

## 📊 Evaluation Matrix
The system scores the user's communication skills based on the following three-pillar matrix:

### 🎙️ Vocal Delivery (Audio)
| Metric | Ideal Target | How We Measure It |
| :--- | :--- | :--- |
| **Speaking Pace** | 130 – 160 WPM | Whisper word count vs. duration |
| **Filler Words** | As close to 0 as possible | Exact text matching on transcript |
| **Pitch Variation** | High variation (Hz) | Librosa frequency analysis |
| **Pauses** | 1+ second gaps | Silence between Whisper segments |

### 🎥 Body Language (Video)
| Metric | Ideal Target | How We Measure It |
| :--- | :--- | :--- |
| **Eye Contact** | Near 100% | MediaPipe Face Detection |
| **Posture** | "Upright" | MediaPipe Pose (Shoulder alignment) |
| **Head Pose** | "Forward" | MediaPipe Face Landmark yaw/pitch |
| **Gestures & Smiles**| Active engagement | Hand tracking and lip curvature |

### 🧠 Linguistic Quality (NLP)
| Metric | Ideal Target | How We Measure It |
| :--- | :--- | :--- |
| **Grammar** | 0 Errors | LanguageTool rules |
| **Vocab Richness** | > 0.40 TTR | NLTK Unique Words ÷ Total Words |
| **Coherence** | > 0.20 | NLTK sentence topic overlap |
| **Readability** | Grade 6 to 8 | Textstat Flesch-Kincaid formula |
