# SpeakX-Pro: Comprehensive System Architecture & Implementation Report

## 1. Executive Overview
SpeakX-Pro is an advanced, privacy-first AI platform engineered to automatically analyze and improve public speaking fluency. The project recently underwent a massive architectural migration from a localized monolithic Streamlit application to a highly scalable, decoupled full-stack architecture comprising a **FastAPI** backend, a **PostgreSQL** database with vector similarity search capabilities, and a responsive **React** frontend.

---

## 2. System Architecture (Microservices Paradigm)

The current state of the application represents a complete separation of concerns:
- **Presentation Layer**: React (Vite) Single Page Application (SPA).
- **Application Layer**: FastAPI (Asynchronous Python Web Framework).
- **Machine Learning Layer**: Python modular pipeline (Whisper, MediaPipe, NLTK).
- **Data & Vector Layer**: PostgreSQL equipped with the `pgvector` extension.

This decoupled architecture allows the Machine Learning pipeline to scale independently from the frontend presentation, solving the inherent blocking issues present in the legacy Streamlit design.

---

## 3. Database Architecture & Schema Deep-Dive

The database was migrated to PostgreSQL using `SQLAlchemy` ORM and `Alembic` for strict version control.

### 3.1. `users` Table (Identity & RBAC)
Handles JSON Web Token (JWT) authentication and Role-Based Access Control.
- `id` (Integer, PK): Unique identifier.
- `username` & `email` (String, Unique): User credentials.
- `password_hash` (Text): Bcrypt-hashed password string.
- `role` (Enum): Enum limiting roles to `student`, `faculty`, `admin`, or `super_admin`.
- `is_active` (Boolean): Soft-delete flag.

### 3.2. `sessions` Table (The Core Metric Store)
Acts as the central repository for all AI-extracted data points from a single video upload.
- **Foreign Keys**: `user_id` links back to `users`.
- **Audio Metrics**: `wpm` (Words Per Minute), `fillers` (Total filler words), `num_pauses`, `avg_pause_sec`, `mean_pitch_hz`, `pitch_variation_hz`.
- **Visual Metrics**: `eye_contact_pct`, `head_pose_dominant`, `posture_upright_pct`, `smile_pct`, `gesture_active_pct`.
- **Linguistic Metrics**: `readability` (Textstat Flesch-Kincaid), `vocabulary_ttr` (Type-Token Ratio), `coherence_score`.
- **Outputs**: `transcript` (Full Whisper text), `llm_report` (Ollama feedback), `overall_score` (Aggregated 0-100 float).

### 3.3. `session_embeddings` Table (Vector Storage)
Powers the AI Chat application.
- `embedding` (Vector 384): Uses `pgvector` to store mathematical representations of the session transcript and feedback.
- **Denormalization Strategy**: Stores `user_id` directly in this table to bypass expensive SQL `JOIN` operations when querying vectors, ensuring `<50ms` vector retrieval times.

### 3.4. `chat_history` Table
Stores conversational memory.
- `role`: Limits to `user` or `assistant`.
- `message`: The raw text content of the message.

---

## 4. Machine Learning & AI Pipeline Deep-Dive

The `/analyze` endpoint triggers a heavily modularized Machine Learning pipeline (`ml_pipeline/`).

### 4.1. Audio & Vocal Processing (`audio_ai.py`)
- **Transcription**: Utilizes OpenAI's `Whisper` model to transcribe speech-to-text. It captures highly accurate timestamps to isolate silent gaps (pauses).
- **Acoustic Profiling**: Utilizes `librosa` to extract the fundamental frequency (F0) of the user's voice to calculate pitch variation, flagging monolithic (robotic) delivery.
- **Filler Word Detection**: Uses regex and NLP token matching to count instances of "um", "ah", "like", and "you know".

### 4.2. Visual & Body Language Processing (`vision_ai.py`)
- **MediaPipe Framework**: Employs Google's MediaPipe Holistic model to process the video frame-by-frame.
- **Eye Contact**: Uses Face Mesh landmarks (specifically the iris/pupil coordinates relative to the eye bounds) to estimate gaze trajectory.
- **Posture**: Tracks shoulder and hip coordinates (`PoseLandmark.LEFT_SHOULDER` etc.) to determine spine alignment and detect slouching.
- **Gestures**: Tracks hand landmarks to measure the bounding-box velocity of the user's hands.

### 4.3. Linguistic & Structural Processing (`text_ai.py`)
- **NLTK**: Performs sentence tokenization and lemmatization to calculate the Type-Token Ratio (unique words divided by total words) to measure vocabulary richness.
- **Textstat**: Calculates the Flesch-Kincaid reading grade level.
- **LanguageTool**: Connects to the local/public LanguageTool API to identify granular grammar and syntax errors.

### 4.4. LLM Synthesis (`llm_ai.py`)
- Takes the 20+ extracted metrics and sends them to a locally hosted Large Language Model via `Ollama`.
- Uses strict prompt engineering to force the LLM to output constructive, personalized coaching feedback in a structured format.

---

## 5. RAG (Retrieval-Augmented Generation) Chatbot

To enable users to converse with their historical data, a custom RAG pipeline was built from scratch.

1. **Embedding Generation**: When a video analysis completes, `sentence-transformers` (`all-MiniLM-L6-v2`) converts the transcript and AI feedback into a 384-dimensional vector.
2. **Storage**: The vector is saved to `pgvector`.
3. **Retrieval**: When a user posts a message to `/api/v1/chat/`, the user's message is vectorized using the exact same transformer model.
4. **Cosine Similarity Search**: The database runs a vector math query (`<->`) to find the top 3 most mathematically similar past session embeddings belonging *only* to that specific `user_id`.
5. **Context Injection**: The retrieved historical data is dynamically injected into the system prompt of the Ollama LLM, enabling it to answer questions like *"Did my eye contact improve since last week?"* accurately.

---

## 6. Frontend Deep-Dive (React + Vite)

The presentation layer is built using Vite, React 18, and TypeScript, styled with Tailwind CSS v3 and `shadcn/ui` components.

### 6.1. Authentication State (`AuthContext.tsx`)
- Uses a global React Context to wrap the application.
- Securely handles the JWT `access_token` returned from the `/login` endpoint.
- Injects the `Authorization: Bearer <token>` header globally into all outgoing `axios` requests via an interceptor.

### 6.2. Theme Provider (`theme-provider.tsx`)
- Modifies the HTML root node classes dynamically.
- Injects a highly customized Glassmorphic Dark Theme configured via CSS variables in `index.css`.

### 6.3. Pages & Routing
- **`Login.tsx`**: Handles form submission and JWT decoding (to extract RBAC roles).
- **`DashboardLayout.tsx`**: A responsive Sidebar/Navbar layout incorporating `lucide-react` icons.
- **`Dashboard.tsx`**: The main landing page. Fetches `GET /api/v1/sessions/`. Utilizes `recharts` to render a LineChart tracking the user's historical `overall_score` progression. Uses dynamic KPI cards for averages.
- **`AnalyzeSession.tsx`**: The Video Upload module. Implements a hidden HTML5 file input wrapped in a styled drag-and-drop zone. Features asynchronous loading states and gracefully handles FastAPI HTTP Exceptions.
- **`AiCoach.tsx`**: The RAG Chat interface. Maintains a local `messages` state array. Features an auto-scrolling effect using React `useRef` tied to the message map length.
