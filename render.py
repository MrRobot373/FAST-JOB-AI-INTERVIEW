# Replacing Firestore with PostgreSQL using SQLAlchemy
import os
import uuid
import json
import logging
from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from dotenv import load_dotenv
from sqlalchemy import create_engine, Column, String, JSON as SQLJSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from google import genai
from google.genai import types
from google.api_core.exceptions import GoogleAPIError

# Load environment variables
load_dotenv()

# Logging setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Gemini API setup
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    logger.critical("GEMINI_API_KEY missing in .env!")
    exit(1)
client = genai.Client(api_key=GEMINI_API_KEY)

# Database setup
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    logger.critical("DATABASE_URL not set in environment!")
    exit(1)
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

class Session(Base):
    __tablename__ = "sessions"
    user_id = Column(String, primary_key=True, index=True)
    session_id = Column(String)
    firstname = Column(String)
    skills = Column(String)
    role = Column(String)
    experience = Column(String)
    history = Column(SQLJSON)

Base.metadata.create_all(bind=engine)

# FastAPI app setup
app = FastAPI(title="AI Interview Bot")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# System instruction prompt
system_instruction = """
You are Alex, a professional interviewer for the {role} position.
Your goal is to ask one concise question at a time in a formal and professional manner.

First ask: \"Tell me about yourself.\"
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
- Do NOT say \"How can I help you?\" or include asterisks.
""".strip()

# Utility functions
def get_or_create_session(user_id, firstname, skills, role, experience):
    db = SessionLocal()
    session_obj = db.query(Session).filter_by(user_id=user_id).first()
    if session_obj:
        session_obj.firstname = firstname
        session_obj.skills = skills
        session_obj.role = role
        session_obj.experience = experience
    else:
        session_obj = Session(
            user_id=user_id,
            session_id=str(uuid.uuid4()),
            firstname=firstname,
            skills=skills,
            role=role,
            experience=experience,
            history=[]
        )
        db.add(session_obj)
    db.commit()
    db.refresh(session_obj)
    db.close()
    return session_obj

# Routes
@app.get("/health")
async def health_check():
    return {"status": "ok"}
    
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
):
    try:
        session_obj = get_or_create_session(user_id, firstname, skills, role, experience)
        chat_history = session_obj.history or []

        contents = [
            types.Content(
                role="assistant",
                parts=[types.Part.from_text(text=system_instruction.format(role=role, firstname=firstname))]
            )
        ]
        for e in chat_history:
            contents.append(types.Content(
                role=e["role"],
                parts=[types.Part.from_text(text=e["text"])]
            ))
        contents.append(types.Content(
            role="user",
            parts=[types.Part.from_text(text=message)]
        ))

        cfg = types.GenerateContentConfig(response_mime_type="text/plain", temperature=0.0, top_p=1.0)

        response_text = ""
        for chunk in client.models.generate_content_stream(
            model="gemini-2.0-flash-lite", contents=contents, config=cfg
        ):
            response_text += chunk.text

        answer = response_text.strip() or "Sorry, I couldn't generate a response. Please try again."

        chat_history.extend([
            {"role": "user", "text": message},
            {"role": "assistant", "text": answer}
        ])

        db = SessionLocal()
        session_obj.history = chat_history
        db.merge(session_obj)
        db.commit()
        db.close()

        return JSONResponse({
            "response": answer,
            "session_id": session_obj.session_id,
            "full_history": chat_history
        })

    except GoogleAPIError as e:
        logger.error(f"Google API Error: {e}", exc_info=True)
        raise HTTPException(status_code=502, detail="Error communicating with AI service.")
    except Exception as e:
        logger.error(f"Talk error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error.")

@app.post("/clear_chat/{user_id}")
async def clear_chat(user_id: str):
    db = SessionLocal()
    session_obj = db.query(Session).filter_by(user_id=user_id).first()
    if session_obj:
        session_obj.history = []
        db.commit()
        db.close()
        return {"message": "Chat history cleared"}
    else:
        db.close()
        raise HTTPException(status_code=404, detail="Session not found")

@app.post("/final_report")
async def final_report(user_id: str = Form(...)):
    db = SessionLocal()
    session_obj = db.query(Session).filter_by(user_id=user_id).first()
    if not session_obj:
        db.close()
        raise HTTPException(status_code=404, detail="Session not found")

    chat_history = session_obj.history or []
    db.close()

    prompt = (
        f"You are a professional technical interviewer.\n\n"
        f"Based on the following conversation history, generate a detailed, Unbiased final interview report in 600 words.\n"
        "The report must include:\n"
        "- Overall Assessment\n"
        "- Strengths\n"
        "- Areas for Improvement\n"
        "- Final Recommendation\n\n"
        "If the interview appears incomplete (e.g., only a few questions answered, poor detail, or short session), "
        "clearly mention this in the report under a section titled 'Interview Incompleteness Notice'. "
        "Avoid assuming unknown qualifications. Do not fabricate answers or assessments.\n\n"
        "Do not include date, time, or placeholder values like 'N/A'. Focus on clarity, conciseness, and professionalism."
    )
    for e in chat_history:
        prompt += f"\n{e['role'].capitalize()}: {e['text']}"

    cfg = types.GenerateContentConfig(response_mime_type="text/plain", temperature=0.0, top_p=1.0)
    resp = client.models.generate_content(
        model="gemini-2.0-flash-lite",
        contents=[types.Content(role="user", parts=[types.Part.from_text(text=prompt)])],
        config=cfg
    )

    report = resp.text.strip() or "Could not generate report."
    return JSONResponse({"report": report})
