import os
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "gcs-key.json"
import uuid
import json
import logging
import asyncio
import re

import edge_tts
from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from celery import Celery
from dotenv import load_dotenv

from google import genai
from google.genai import types
from google.api_core.exceptions import GoogleAPIError
from google.cloud import firestore

# Initialize Firestore client
db = firestore.Client()

from pathlib import Path

# ——— env & logging ———
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    logger.critical("GEMINI_API_KEY missing in .env!") and exit(1)
client = genai.Client(api_key=GEMINI_API_KEY)

CELERY_BROKER_URL = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")

# ——— Celery TTS task ———
celery_app = Celery("main", broker=CELERY_BROKER_URL)
celery_app.conf.broker_connection_retry_on_startup = True
VOICE = "en-CA-LiamNeural"
BASE_DIR = Path(__file__).resolve().parent


from google.cloud import storage
from datetime import timedelta

# Initialize GCS client
storage_client = storage.Client()
bucket_name = "ai-interview-audio-nihar10100"  # Replace with your actual bucket name

@celery_app.task(bind=True, max_retries=3, default_retry_delay=300)
def generate_tts_task(self, text: str, user_id: str, session_id: str):
    def clean_tts_text(t: str) -> str:
        t = t.replace("“", '"').replace("”", '"')
        t = t.replace("‘", "'").replace("’", "'")
        return re.sub(r'\s+', ' ', t).strip()

    logger.info(f"[TTS] Starting TTS task for session_id={session_id}")
    
    cleaned = clean_tts_text(text)
    tmp_path = Path(f"/tmp/response_{session_id}.mp3.tmp")

    try:
        asyncio.run(edge_tts.Communicate(cleaned, VOICE).save(str(tmp_path)))
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(f"tts_audio/response_{session_id}.mp3")
        blob.upload_from_filename(str(tmp_path))
        tmp_path.unlink()

        doc_ref = db.collection("sessions").document(session_id)
        if doc_ref.get().exists:
            doc_ref.update({"audio_ready": True})
            logger.info(f"[TTS] Uploaded audio and updated Firestore for session {session_id}")
        else:
            logger.warning(f"[TTS] No document found with session_id: {session_id}")

    except Exception as e:
        logger.error(f"[TTS] Error for session {session_id}: {e}", exc_info=True)
        try:
            self.retry(exc=e)
        except self.MaxRetriesExceededError:
            logger.error(f"[TTS] Max retries exceeded for session {session_id}")



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

# In-memory store for chat sessions (for demonstration purposes without a DB)
# In a real application, you'd integrate with a persistent store (e.g., another microservice's DB)
chat_sessions = {}

def get_or_create_session(user_id, firstname, skills, role, experience):
    # Search for session by user_id (if needed)
    sessions = db.collection("sessions").where("user_id", "==", user_id).limit(1).stream()
    session_doc = next(sessions, None)

    if session_doc:
        session = session_doc.to_dict()
        session.update({
            "firstname": firstname,
            "skills": skills,
            "role": role,
            "experience": experience
        })
        doc_ref = db.collection("sessions").document(session["id"])
        doc_ref.set(session, merge=True)
        logger.info(f"Updated Firestore session {session['id']} for user {user_id}")
        return session
    else:
        session_id = str(uuid.uuid4())
        session = {
            "id": session_id,
            "user_id": user_id,
            "firstname": firstname,
            "skills": skills,
            "role": role,
            "experience": experience,
            "history": [],
            "audio_ready": False
        }
        doc_ref = db.collection("sessions").document(session_id)
        doc_ref.set(session)
        logger.info(f"Created Firestore session {session_id} for user {user_id}")
        return session



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
        session = get_or_create_session(user_id, firstname, skills, role, experience)
        chat_history = session.get("history", [])

        # build contents for Gemini
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

       # Update chat history
        chat_history.extend([
            {"role": "user", "text": message},
            {"role": "assistant", "text": answer}
        ])
        session["history"] = chat_history
        
        doc_ref = db.collection("sessions").document(session["id"])
        doc_ref.update({"audio_ready": False})
        generate_tts_task.delay(answer, user_id, session["id"])


        return JSONResponse({
            "response": answer,
            "tts_status": "processing",
            "session_id": session["id"],
            "full_history": chat_history,
        })

    except GoogleAPIError as e:
        logger.error(f"Google API Error: {e}", exc_info=True)
        raise HTTPException(status_code=502, detail="Error communicating with AI service.")
    except Exception as e:
        logger.error(f"Talk error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error.")

from datetime import timedelta

@app.get("/get_audio/{session_id}")
async def get_audio(session_id: str):
    # Use session_id directly as the Firestore document ID
    doc_ref = db.collection("sessions").document(session_id)
    doc = doc_ref.get()

    if not doc.exists:
        raise HTTPException(status_code=404, detail="Session not found")

    session_data = doc.to_dict()
    if not session_data.get("audio_ready"):
        raise HTTPException(status_code=404, detail="Audio not ready yet")

    blob = storage_client.bucket(bucket_name).blob(f"tts_audio/response_{session_id}.mp3")
    signed_url = blob.generate_signed_url(
        version="v4",
        expiration=timedelta(hours=1),
        method="GET"
    )
    return JSONResponse({"audio_url": signed_url})


@app.post("/clear_chat/{session_id}")
async def clear_chat(session_id: str):
    doc_ref = db.collection("sessions").document(session_id)
    doc = doc_ref.get()
    if doc.exists:
        doc_ref.update({"history": []})
        return {"message": "Chat history cleared"}
    else:
        raise HTTPException(status_code=404, detail="Session not found")


@app.post("/final_report")
async def final_report(session_id: str = Form(...)):
    doc_ref = db.collection("sessions").document(session_id)
    doc = doc_ref.get()

    if not doc.exists:
        raise HTTPException(status_code=404, detail="Session not found")

    session = doc.to_dict()
    chat_history = session.get("history", [])

    # Create prompt for the report generation
    prompt = (
        f"Generate a final interview report for {session['firstname']} "
        f"applying for {session['role']} with {session['experience']} years experience "
        f"and skills in {session['skills']}.\n\n"
        "Include: Overall Assessment, Strengths, Areas for Improvement, and Recommendation.\n\n"
    )
    
    # Add chat history to the prompt
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
