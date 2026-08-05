import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from db.session import SessionLocal
from db.models import User, UserRole
from core.security import get_password_hash

def seed_users():
    db = SessionLocal()
    if not db.query(User).first():
        admin = User(
            username="admin",
            email="admin@speakx.com",
            password_hash=get_password_hash("admin123"),
            role=UserRole.super_admin
        )
        student = User(
            username="student1",
            email="student1@speakx.com",
            password_hash=get_password_hash("student123"),
            role=UserRole.student
        )
        db.add(admin)
        db.add(student)
        db.commit()
        print("Database seeded with default users (admin / student1).")
    else:
        print("Database already contains users. Skipping seed.")
    db.close()

if __name__ == "__main__":
    seed_users()
