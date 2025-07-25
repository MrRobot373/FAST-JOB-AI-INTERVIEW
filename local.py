# for local with mysql database (no gcp connect)
import os
import uuid
import json
import logging
import asyncio
import re

import edge_tts
from fastapi import FastAPI, Form, Depends, HTTPException, Request
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from sqlalchemy import create_engine, Column, String, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session

from celery import Celery
from dotenv import load_dotenv

from google import genai
from google.genai import types
from google.api_core.exceptions import GoogleAPIError
from pathlib import Path

# ——— env & logging ———
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    logger.critical("GEMINI_API_KEY missing in .env!") and exit(1)
client = genai.Client(api_key=GEMINI_API_KEY)

CELERY_BROKER_URL = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
MYSQL_DATABASE_URL = os.getenv(
    "MYSQL_DATABASE_URL",
    "mysql+pymysql://root:1234@localhost/interviewbot"
)

# ——— DB setup ———
engine = create_engine(MYSQL_DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
Base = declarative_base()

class ChatSession(Base):
    __tablename__ = "chat_sessions"
    id = Column(String(64), primary_key=True, index=True, unique=True)
    user_id = Column(String(64), index=True, nullable=False)
    firstname = Column(String(64), nullable=False)
    skills = Column(Text, nullable=False)
    role = Column(String(64), nullable=False)
    experience = Column(String(32), nullable=False)
    history = Column(Text, default="[]")

Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try: yield db
    finally: db.close()

def get_or_create_session(db: Session, user_id, firstname, skills, role, experience):
    session = db.query(ChatSession).filter_by(user_id=user_id).first()
    if not session:
        session = ChatSession(
            id=str(uuid.uuid4()),
            user_id=user_id,
            firstname=firstname,
            skills=skills,
            role=role,
            experience=experience,
            history="[]"
        )
        db.add(session)
        db.commit()
        db.refresh(session)
        logger.info(f"Created session {session.id} for user {user_id}")
    else:
        session.firstname, session.skills, session.role, session.experience = (
            firstname, skills, role, experience
        )
        db.commit()
        db.refresh(session)
        logger.info(f"Updated session {session.id} for user {user_id}")
    return session

# ——— Celery TTS task ———
celery_app = Celery("main", broker=CELERY_BROKER_URL)
VOICE = "en-CA-LiamNeural"
BASE_DIR = Path(__file__).resolve().parent
TEMP_DIR = BASE_DIR / "temp"
TEMP_DIR.mkdir(exist_ok=True)

@celery_app.task(bind=True, max_retries=3, default_retry_delay=300)
def generate_tts_task(self, text: str, session_id: str):
    def clean_tts_text(t: str) -> str:
        t = t.replace("“", '"').replace("”", '"')
        t = t.replace("‘", "'").replace("’", "'")
        return re.sub(r'\s+', ' ', t).strip()
    cleaned = clean_tts_text(text)
    tmp_path = TEMP_DIR / f"response_{session_id}.mp3.tmp"
    final_path = TEMP_DIR / f"response_{session_id}.mp3"
    if final_path.exists():
        final_path.unlink()
        logger.info(f"[TTS] removed old → {final_path}")
    async def run_tts():
        await edge_tts.Communicate(cleaned, VOICE).save(str(tmp_path))
        tmp_path.replace(final_path)
        logger.info(f"[TTS] wrote & replaced → {final_path}")
    try:
        asyncio.run(run_tts())
    except Exception as e:
        logger.error(f"[TTS] error: {e}", exc_info=True)
        try: self.retry(exc=e)
        except self.MaxRetriesExceededError:
            logger.error(f"[TTS] max retries for {session_id}")

# ——— FastAPI setup ———
app = FastAPI(title="AI Interview Bot")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

system_instruction = """
You are Alex, a professional interviewer for the {role} position.
Your goal is to ask one concise question at a time in a formal and professional manner.

First ask: "Tell me about yourself."
Then follow up with 10–15 structured verbal questions covering technical skills, problem-solving,
real-world scenarios, and debugging. Keep it professional, avoid repetitive patterns,
include scenario-based and logic questions, and do not ask coding questions.
Conclude by informing {firstname} that results will arrive via email and provide a final evaluation.

Behavior rules:
- Stay in character: Always remain a formal interviewer.
- Ask only ONE question at a time.
- Do NOT provide answers or hints.
- If an answer is incomplete, ask for more details.
- If the candidate doesn’t know, move on without offering solutions.
- Do NOT say "How can I help you?" or include asterisks.
""".strip()

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/talk")
async def talk(
    user_id: str = Form(...),
    firstname: str = Form(...),
    skills: str = Form(...),
    role: str = Form(...),
    experience: str = Form(...),
    message: str = Form(...),
    db: Session = Depends(get_db),
):
    try:
        session = get_or_create_session(db, user_id, firstname, skills, role, experience)
        chat_history = json.loads(session.history or "[]")

        # build contents
        contents = [
            types.Content(
                role="assistant",  # Gemini only supports user & assistant roles
                parts=[types.Part.from_text(text=system_instruction.format(role=role, firstname=firstname))]
            )
        ]
        for e in chat_history:
            api_role = "user" if e["role"] == "user" else "assistant"
            contents.append(types.Content(
                role=api_role,
                parts=[types.Part.from_text(text=e["text"])]
            ))
        contents.append(types.Content(
            role="user",
            parts=[types.Part.from_text(text=message)]
        ))

        cfg = types.GenerateContentConfig(
            response_mime_type="text/plain",
            temperature=0.0,
            top_p=1.0
        )

        response_text = ""
        for chunk in client.models.generate_content_stream(
            model="gemini-2.0-flash-lite",
            contents=contents,
            config=cfg
        ):
            response_text += chunk.text

        answer = response_text.strip() or "Sorry, I couldn't generate a response. Please try again."

        # save history
        chat_history.extend([
            {"role": "user", "text": message},
            {"role": "assistant", "text": answer}
        ])
        session.history = json.dumps(chat_history)
        db.commit()
        db.refresh(session)

        # dispatch TTS
        generate_tts_task.delay(answer, session.id)

        return JSONResponse({
            "response": answer,
            "tts_status": "processing",
            "session_id": session.id,
            "full_history": chat_history,
        })

    except GoogleAPIError as e:
        logger.error(f"Google API Error: {e}", exc_info=True)
        raise HTTPException(status_code=502, detail="Error communicating with AI service.")
    except Exception as e:
        logger.error(f"Talk error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error.")

@app.get("/get_audio/{session_id}")
async def get_audio(session_id: str):
    f = TEMP_DIR / f"response_{session_id}.mp3"
    if f.exists():
        return FileResponse(str(f), media_type="audio/mpeg", filename=f.name)
    raise HTTPException(status_code=404, detail="Audio not ready.")

@app.post("/clear_chat/{user_id}")
async def clear_chat(user_id: str, db: Session = Depends(get_db)):
    sess = db.query(ChatSession).filter_by(user_id=user_id).first()
    if sess:
        sess.history = "[]"
        db.commit()
        return {"message": "Chat history cleared"}
    raise HTTPException(status_code=404, detail="Session not found")

@app.post("/final_report")
async def final_report(user_id: str = Form(...), db: Session = Depends(get_db)):
    session = db.query(ChatSession).filter_by(user_id=user_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    chat_history = json.loads(session.history or "[]")
    prompt = (
        f"Generate a final interview report in 300 words for {session.firstname} "
        f"applying for {session.role} with {session.experience} years experience "
        f"and skills in {session.skills}.\n\n"
        "Include: Overall Assessment, Strengths, Areas for Improvement, Recommendation.\n\n"
    )
    for e in chat_history:
        prompt += f"{e['role'].capitalize()}: {e['text']}\n"

    cfg = types.GenerateContentConfig(
        response_mime_type="text/plain",
        temperature=0.0,
        top_p=1.0
    )
    resp = client.models.generate_content(
        model="gemini-2.0-flash-lite",
        contents=[types.Content(role="user", parts=[types.Part.from_text(text=prompt)])],
        config=cfg
    )
    report = resp.text.strip() or "Could not generate report."
    return JSONResponse({"report": report})
