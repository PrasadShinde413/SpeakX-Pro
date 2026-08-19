from datetime import timedelta
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from pydantic import BaseModel

from db.session import get_db
from db.models import User, UserRole
from core.security import verify_password, create_access_token, ACCESS_TOKEN_EXPIRE_MINUTES
from api.deps import get_current_active_user, require_role

router = APIRouter()

class Token(BaseModel):
    access_token: str
    token_type: str
    role: str

class UserResponse(BaseModel):
    id: int
    username: str
    email: str
    role: str
    
    class Config:
        from_attributes = True

@router.post("/login", response_model=Token)
def login_for_access_token(
    form_data: OAuth2PasswordRequestForm = Depends(), 
    db: Session = Depends(get_db)
):
    # Authenticate user
    user = db.query(User).filter(User.email == form_data.username).first() # frontend uses email as username field
    
    if not user and "@" not in form_data.username:
        # Fallback to username just in case
        user = db.query(User).filter(User.username == form_data.username).first()
        
    if not user or not verify_password(form_data.password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username/email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if not user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")

    # Generate JWT
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username, "role": user.role.value}, expires_delta=access_token_expires
    )
    
    return {
        "access_token": access_token, 
        "token_type": "bearer",
        "role": user.role.value
    }

@router.get("/me", response_model=UserResponse)
def read_users_me(current_user: User = Depends(get_current_active_user)):
    """
    Returns the currently logged-in user.
    """
    return current_user

@router.get("/admin-only", response_model=dict)
def test_admin_route(current_user: User = Depends(require_role([UserRole.admin, UserRole.super_admin]))):
    """
    Test route to verify RBAC is working. Only admins can hit this.
    """
    return {"message": f"Welcome Admin {current_user.username}!"}
