import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

# The default URL matches the docker-compose.yml configuration
DATABASE_URL = os.environ.get(
    "DATABASE_URL", 
    "postgresql://speakx_admin:speakx_password@127.0.0.1:5433/speakx_pro"
)

# Create the SQLAlchemy engine
engine = create_engine(DATABASE_URL)

# Create a session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for declarative models
Base = declarative_base()

# Dependency to get the DB session in FastAPI
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
