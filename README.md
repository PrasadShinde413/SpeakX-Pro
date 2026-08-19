# 🎥 SpeakX-Pro (Video Analyzer)

**SpeakX-Pro** is an intelligent, privacy-first AI platform designed to help individuals dramatically improve their public speaking skills, communication fluency, and self-confidence through the automated analysis of video recordings.

This project has been completely rebuilt from a monolithic Streamlit app into a modern, full-stack, decoupled architecture utilizing **FastAPI**, **PostgreSQL (`pgvector`)**, and a sleek **React (`shadcn/ui`)** frontend.

---

## 🎯 Architecture Overview

The platform is divided into three core pillars:

1. **The Machine Learning Pipeline (Python)**
   - Extracts over 20 unique vocal, visual, and linguistic data points from user videos.
   - **Audio/Vocal:** Whisper (transcription, pace), Librosa (pitch, variation, pauses).
   - **Visual/Body Language:** MediaPipe (eye contact, posture, gestures, head pose).
   - **NLP/Linguistics:** NLTK, Textstat, LanguageTool (grammar, coherence, readability).
   - **RAG Embeddings:** `sentence-transformers` automatically convert the analysis results into 384-dimensional vectors.

2. **The Backend API (FastAPI & PostgreSQL)**
   - Exposes RESTful endpoints for Video Upload (`/analyze`), Authentication (`/login`), and RAG Chat (`/chat`).
   - Uses **JWT Authentication** for secure Role-Based Access Control (RBAC). Admin users can see all videos, while Students can only see their own.
   - Stores session data in a **PostgreSQL** database.
   - Uses the **`pgvector`** extension to run natively hardware-accelerated cosine similarity searches across past AI coaching feedback for the AI Chat feature!

3. **The Frontend Dashboard (React + Vite)**
   - A stunning, glassmorphic UI built using **Tailwind CSS** and **shadcn/ui**.
   - Features a Light/Dark Mode toggle.
   - Includes a visual "Student Dashboard" featuring interactive **Recharts** graphs to track performance trends over time.
   - Provides a ChatGPT-style conversational UI to chat with your personalized AI speaking coach.

---

## 🚀 How to Run Locally

Because the application is fully decoupled, you must run both the Backend and Frontend servers concurrently.

### 1. Prerequisites
- Python 3.10+
- Node.js (v18+)
- PostgreSQL installed locally with the `pgvector` extension enabled.
- [Ollama](https://ollama.com/) running in the background with your preferred LLM (e.g., `ollama run qwen2.5`).

### 2. Run the Backend (FastAPI)
Open your first terminal and run:
```bash
cd "Video Analyzer/Backend"

# Activate your virtual environment
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Mac/Linux

# Start the server on http://127.0.0.1:8000
uvicorn main:app --reload
```

### 3. Run the Frontend (React)
Open a second terminal and run:
```bash
cd "Video Analyzer/frontend"

# Ensure dependencies are installed
npm install

# Start the Vite development server
npm run dev
```

Navigate your browser to **http://localhost:5173** and log in!
- **Student Demo:** `student1@speakx.com` / `student123`
- **Admin Demo:** `admin@speakx.com` / `admin123`

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
