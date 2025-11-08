# src/security.py
import os
import hmac
from typing import List, Optional
from fastapi import HTTPException, Request, status, Header
from fastapi.security.api_key import APIKeyHeader
from starlette.status import HTTP_403_FORBIDDEN, HTTP_401_UNAUTHORIZED

# Name of header to expect
API_KEY_HEADER_NAME = "X-API-Key"

# Read admin key(s) from environment variable.
# Support multiple keys (comma-separated) for key-rotation.
# Example: export ADMIN_API_KEYS="key1,key2"
_admin_keys_env = os.getenv("ADMIN_API_KEYS", "")
ADMIN_API_KEYS: List[str] = [k.strip() for k in _admin_keys_env.split(",") if k.strip()]

# Optional IP allowlist (comma-separated) - if non-empty, only these IPs allowed
_ip_allowlist_env = os.getenv("ADMIN_IP_ALLOWLIST", "")
ADMIN_IP_ALLOWLIST: List[str] = [ip.strip() for ip in _ip_allowlist_env.split(",") if ip.strip()]

# FastAPI security dependency (header)
api_key_header = APIKeyHeader(name=API_KEY_HEADER_NAME, auto_error=False)


def _is_key_valid(provided_key: str) -> bool:
    """Constant-time comparison against allowed admin keys."""
    if not provided_key:
        return False
    for k in ADMIN_API_KEYS:
        # hmac.compare_digest to avoid timing attacks
        if hmac.compare_digest(provided_key, k):
            return True
    return False


def _client_ip_from_request(request: Request) -> Optional[str]:
    """
    Extract client IP from request. If behind reverse proxy, ensure the proxy
    sets X-Forwarded-For header and your proxy is configured to pass it.
    """
    xff = request.headers.get("x-forwarded-for")
    if xff:
        # X-Forwarded-For may contain a comma-separated list; take first
        return xff.split(",")[0].strip()
    # Fallback to client.host (Starlette Request.client)
    client = request.client
    if client:
        return client.host
    return None


def verify_admin(request: Request, x_api_key: str = Header(None)):
    allowed_keys = os.getenv("ADMIN_API_KEYS", "").split(",")
    allowed_ips = os.getenv("ADMIN_IP_ALLOWLIST", "").split(",")

    client_ip = request.client.host

    if allowed_ips and client_ip not in allowed_ips:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied: IP not allowed")

    if x_api_key not in allowed_keys:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Invalid API key")

    return True
