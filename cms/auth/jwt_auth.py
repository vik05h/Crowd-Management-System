"""Role-Based Access Control and JWT Authentication for CMS.

Supports roles: 'admin', 'operator', 'viewer'.
Enforces token validation and permissions across security endpoints.
"""

import os
import time
from typing import Annotated, Any, Dict, List, Optional

import jwt
from fastapi import Cookie, Depends, Header, HTTPException, status
from pydantic import BaseModel

from cms.data.database import db_manager, verify_password

# Authentication configuration
JWT_SECRET = os.getenv("CMS_JWT_SECRET", "cms-spatial-safety-secret-key-2026-secure")
JWT_ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_SECONDS = 86400 * 7  # 7 days


class TokenPayload(BaseModel):
    sub: str
    role: str
    exp: int


class LoginRequest(BaseModel):
    username: str
    password: str


class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    role: str
    username: str


def create_access_token(username: str, role: str, expires_in: int = ACCESS_TOKEN_EXPIRE_SECONDS) -> str:
    """Generate signed JWT access token with role claims."""
    expire = int(time.time()) + expires_in
    payload = {
        "sub": username,
        "role": role,
        "exp": expire,
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def decode_access_token(token: str) -> Optional[Dict[str, Any]]:
    """Decode and validate a signed JWT token."""
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except (jwt.PyJWTError, Exception):
        return None


async def get_current_user_optional(
    authorization: Annotated[Optional[str], Header()] = None,
    cms_token: Annotated[Optional[str], Cookie()] = None,
) -> Optional[Dict[str, Any]]:
    """Extract authenticated user if token is present, else None."""
    token = None
    if authorization and authorization.startswith("Bearer "):
        token = authorization[7:].strip()
    elif cms_token:
        token = cms_token.strip()

    if not token:
        return None

    payload = decode_access_token(token)
    if not payload:
        return None

    username = payload.get("sub")
    if not username:
        return None

    user = await db_manager.get_user_by_username(username)
    if not user:
        return None

    return {
        "id": user["id"],
        "username": user["username"],
        "role": user["role"],
    }


async def get_current_user(
    current_user: Annotated[Optional[Dict[str, Any]], Depends(get_current_user_optional)],
) -> Dict[str, Any]:
    """Require valid authenticated user."""
    if not current_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication credentials required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return current_user


CurrentUserDep = Annotated[Dict[str, Any], Depends(get_current_user)]
OptionalUserDep = Annotated[Optional[Dict[str, Any]], Depends(get_current_user_optional)]


def require_roles(allowed_roles: List[str]):
    """Dependency factory checking if current user holds an authorized role."""

    async def role_checker(user: CurrentUserDep) -> Dict[str, Any]:
        if user["role"] not in allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Access denied. Requires one of roles: {', '.join(allowed_roles)}",
            )
        return user

    return role_checker
