import enum
from sqlalchemy import Column, Integer, String, Float, Text, Boolean, DateTime, ForeignKey, Enum as SQLEnum
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from pgvector.sqlalchemy import Vector
from .session import Base

class UserRole(enum.Enum):
    student = "student"
    faculty = "faculty"
    admin = "admin"
    super_admin = "super_admin"

class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, nullable=False, index=True)
    email = Column(String(150), unique=True, nullable=False, index=True)
    password_hash = Column(Text, nullable=False)
    role = Column(SQLEnum(UserRole), nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, server_default=func.now())
    
    sessions = relationship("Session", back_populates="user", cascade="all, delete-orphan")
    chat_history = relationship("ChatHistory", back_populates="user", cascade="all, delete-orphan")

class Session(Base):
    __tablename__ = "sessions"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    session_date = Column(DateTime, server_default=func.now(), index=True)
    video_filename = Column(Text)
    duration_sec = Column(Float)
    
    # Audio Metrics
    wpm = Column(Integer)
    fillers = Column(Integer)
    fillers_per_minute = Column(Float)
    num_pauses = Column(Integer)
    avg_pause_sec = Column(Float)
    total_pause_sec = Column(Float)
    mean_pitch_hz = Column(Float)
    pitch_variation_hz = Column(Float)
    
    # Video Metrics
    eye_contact_pct = Column(Float)
    head_pose_forward_pct = Column(Float)
    head_pose_dominant = Column(String(20))
    posture_upright_pct = Column(Float)
    posture_dominant = Column(String(20))
    gesture_active_pct = Column(Float)
    smile_pct = Column(Float)
    
    # NLP Metrics
    readability = Column(String(50))
    grade_level = Column(String(20))
    reading_ease = Column(Float)
    vocabulary_ttr = Column(Float)
    coherence_score = Column(Float)
    total_sentences = Column(Integer)
    avg_sentence_length = Column(Float)
    
    # Full Text & Outcomes
    transcript = Column(Text)
    llm_report = Column(Text)
    overall_score = Column(Float)
    
    user = relationship("User", back_populates="sessions")
    embeddings = relationship("SessionEmbedding", back_populates="session", cascade="all, delete-orphan")

class SessionEmbedding(Base):
    __tablename__ = "session_embeddings"
    
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False)
    # Denormalized user_id to make RBAC queries fast without needing a SQL JOIN
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    
    content = Column(Text, nullable=False)
    embedding = Column(Vector(384), nullable=False) # 384 dims for sentence-transformers/all-MiniLM-L6-v2
    
    session = relationship("Session", back_populates="embeddings")

class ChatHistory(Base):
    __tablename__ = "chat_history"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    role = Column(String(10), nullable=False) # 'user' or 'assistant'
    message = Column(Text, nullable=False)
    created_at = Column(DateTime, server_default=func.now(), index=True)
    
    user = relationship("User", back_populates="chat_history")
