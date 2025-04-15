from base64 import b64decode, b64encode
import json
import os
from pathlib import Path
from time import time

from authlib.integrations.starlette_client import OAuth
from cryptography.fernet import Fernet
from dotenv import load_dotenv
from fastapi import Request
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
import google.adk.cli.fast_api
from google.adk.cli.fast_api import get_fast_api_app
from starlette.middleware.sessions import SessionMiddleware
import uvicorn

import global_var


load_dotenv()

OPENID_CLIENT_ID = os.environ["OPENID_CLIENT_ID"]
OPENID_CLIENT_SECRET = os.environ["OPENID_CLIENT_SECRET"]
SECRET_KEY = os.environ["SECRET_KEY"]
SYMMETRIC_KEY = os.environ["SYMMETRIC_KEY"]

BASE_PATH = Path(google.adk.cli.fast_api.__file__).parent.resolve()
ANGULAR_DIST_PATH = BASE_PATH / "browser"

symmetric_key = Fernet(SYMMETRIC_KEY)

app = get_fast_api_app(
    agent_dir=Path(".").resolve(),
    session_db_url="sqlite:///database.db",
    web=False,
)

oauth = OAuth()
oauth.register(
    "google",
    client_id=OPENID_CLIENT_ID,
    client_secret=OPENID_CLIENT_SECRET,
    userinfo_endpoint="https://openidconnect.googleapis.com/v1/userinfo",
    server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
    client_kwargs={"scope": "https://www.googleapis.com/auth/calendar email openid"},
)


def validate_token(request: Request, encrypted_token: str) -> bool:
    try:
        token = json.loads(symmetric_key.decrypt(b64decode(encrypted_token)).decode())
        if token["exp"] < int(time()):
            return False
        request.state.token = token
        return True
    except Exception:
        return False


@app.middleware("http")
async def store_token(request: Request, call_next):
    if not hasattr(request.state, "token"):
        return await call_next(request)
    global_var.var.set(request.state.token)
    res = await call_next(request)
    global_var.var.set(None)
    return res


@app.middleware("http")
async def authenticate(request: Request, call_next):
    encrypted_token = request.session.get("token")
    if not request.url.path.startswith("/auth") and (
        encrypted_token is None or not validate_token(request, encrypted_token)
    ):
        return await oauth.google.authorize_redirect(
            request, "http://localhost:8001/auth"
        )
    return await call_next(request)


app.add_middleware(SessionMiddleware, secret_key=SECRET_KEY)


@app.get("/")
async def index(request: Request):
    return RedirectResponse("/dev-ui")


@app.get("/auth")
async def auth(request: Request):
    res = await oauth.google.authorize_access_token(request)
    res["exp"] = int(time()) + res["expires_in"]
    request.session["token"] = b64encode(
        symmetric_key.encrypt(json.dumps(res).encode())
    ).decode()
    return RedirectResponse("/dev-ui")


@app.get("/dev-ui")
async def dev_ui():
    return FileResponse(BASE_PATH / "browser/index.html")


app.mount("/", StaticFiles(directory=ANGULAR_DIST_PATH, html=True), name="static")

uvicorn.run(app, port=8001)
