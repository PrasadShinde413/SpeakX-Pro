import os
import tempfile
import sys
from fastapi import APIRouter, Depends, UploadFile, File, HTTPException
from sqlalchemy.orm import Session

# Add root directory to python path so we can import ml_pipeline
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ml_pipeline.audio_ai import analyze_audio
from ml_pipeline.video_ai import analyze_video
from ml_pipeline.nlp_ai import analyze_nlp
from ml_pipeline.llm_ai import generate_feedback
from ml_pipeline.scoring_ai import calculate_scores
from ml_pipeline.embedding_ai import generate_embedding

from db.session import get_db
from db.models import User, Session as SessionModel, SessionEmbedding
from api.deps import get_current_active_user

router = APIRouter()

@router.post("/analyze")
async def analyze_session(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Accepts a video upload, runs the ML pipeline, and stores the session results.
    """
    if not file.filename.endswith(('.mp4', '.mov', '.avi')):
        raise HTTPException(status_code=400, detail="Invalid video format")
        
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_video:
        content = await file.read()
        temp_video.write(content)
        temp_video_path = temp_video.name
        
    try:
        # Run ML Pipeline
        audio_results = analyze_audio(temp_video_path)
        nlp_results = analyze_nlp(audio_results.get("transcript", ""))
        video_results = analyze_video(temp_video_path)
        
        # Calculate scores and generate feedback
        scores = calculate_scores(audio_results, video_results, nlp_results)
        feedback_report = generate_feedback(audio_results, video_results, nlp_results)
        
        # Create database record
        new_session = SessionModel(
            user_id=current_user.id,
            video_filename=file.filename,
            duration_sec=audio_results.get("duration_sec", 0),
            
            wpm=audio_results.get("wpm", 0),
            fillers=audio_results.get("fillers", 0),
            fillers_per_minute=audio_results.get("fillers_per_minute", 0),
            num_pauses=audio_results.get("num_pauses", 0),
            avg_pause_sec=audio_results.get("avg_pause_sec", 0),
            total_pause_sec=audio_results.get("total_pause_sec", 0),
            mean_pitch_hz=audio_results.get("mean_pitch_hz", 0),
            pitch_variation_hz=audio_results.get("pitch_variation_hz", 0),
            
            eye_contact_pct=video_results.get("eye_contact_pct", 0),
            head_pose_forward_pct=video_results.get("head_pose_forward_pct", 0),
            head_pose_dominant=video_results.get("head_pose_dominant", "Unknown"),
            posture_upright_pct=video_results.get("posture_upright_pct", 0),
            posture_dominant=video_results.get("posture_dominant", "Unknown"),
            gesture_active_pct=video_results.get("gesture_active_pct", 0),
            smile_pct=video_results.get("smile_pct", 0),
            
            readability=nlp_results.get("readability", "Unknown"),
            grade_level=str(nlp_results.get("flesch_kincaid_grade", "N/A")),
            reading_ease=nlp_results.get("flesch_reading_ease", 0),
            vocabulary_ttr=nlp_results.get("vocabulary_ttr", 0),
            coherence_score=nlp_results.get("coherence_score", 0),
            total_sentences=nlp_results.get("sentence_count", 0),
            avg_sentence_length=nlp_results.get("avg_sentence_length", 0),
            
            transcript=audio_results.get("transcript", ""),
            llm_report=feedback_report,
            overall_score=scores.get("Overall Performance", 0)
        )
        
        db.add(new_session)
        db.commit()
        db.refresh(new_session)
        
        # --- RAG Integration: Generate and save embedding ---
        combined_text = f"Transcript:\n{audio_results.get('transcript', '')}\n\nFeedback:\n{feedback_report}"
        embedding_vector = generate_embedding(combined_text)
        
        new_embedding = SessionEmbedding(
            session_id=new_session.id,
            user_id=current_user.id,
            embedding=embedding_vector
        )
        db.add(new_embedding)
        db.commit()
        
    finally:
        os.remove(temp_video_path)
        
    return {"message": "Analysis complete", "session_id": new_session.id}

@router.get("/")
def get_user_sessions(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    sessions = db.query(SessionModel).filter(SessionModel.user_id == current_user.id).order_by(SessionModel.session_date.desc()).all()
    return sessions
