# Final code : GCP and production ready (BEFORE DEVOPS) 
import os
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "gcs-key.json"
import uuid
import json
import logging
import asyncio
import re

#import edge_tts
from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

#from celery import Celery
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

# CELERY_BROKER_URL = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")

# # ——— Celery TTS task ———
# celery_app = Celery("main", broker=CELERY_BROKER_URL)
# celery_app.conf.broker_connection_retry_on_startup = True
# VOICE = "en-CA-LiamNeural"
# BASE_DIR = Path(__file__).resolve().parent

# from google.cloud import storage
# from datetime import timedelta

# Initialize GCS client
# storage_client = storage.Client()
# bucket_name = "ai-interview-audio-nihar10100"  # Replace with your actual bucket name

# @celery_app.task(bind=True, max_retries=3, default_retry_delay=300)
# def generate_tts_task(self, text: str, user_id: str, session_id: str):
#     import time  # ensure this is at the top of your file

#     def clean_tts_text(t: str) -> str:
#         t = t.replace("“", '"').replace("”", '"')
#         t = t.replace("‘", "'").replace("’", "'")
#         return re.sub(r'\s+', ' ', t).strip()

#     try:
#         cleaned = clean_tts_text(text)
#         audio_id = f"{session_id}_{int(time.time())}"  # ✅ unique ID per audio
#         tmp_path = Path(f"/tmp/response_{audio_id}.mp3.tmp")

#         # Generate TTS
#         asyncio.run(edge_tts.Communicate(cleaned, VOICE).save(str(tmp_path)))

#         # Upload to GCS with correct audio_id
#         bucket = storage_client.bucket(bucket_name)
#         blob = bucket.blob(f"tts_audio/response_{audio_id}.mp3")  # ✅ uses full audio_id
#         blob.upload_from_filename(str(tmp_path))
#         tmp_path.unlink()

#         # Update Firestore with audio_id and audio_ready flag
#         sessions = db.collection("sessions").where("id", "==", session_id).limit(1).stream()
#         session_doc = next(sessions, None)

#         if session_doc:
#             doc_ref = db.collection("sessions").document(session_doc.id)
#             doc_ref.update({
#                 "last_audio_id": audio_id  # ✅ this must match uploaded file
#             })
#             logger.info(f"[TTS] Uploaded and marked ready: {audio_id}")
#         else:
#             logger.warning(f"[TTS] Session with id {session_id} not found.")

#     except Exception as e:
#         logger.error(f"[TTS] Error during session {session_id}: {e}", exc_info=True)
#         try:
#             self.retry(exc=e)
#         except self.MaxRetriesExceededError:
#             logger.error(f"[TTS] Max retries exceeded for session {session_id}")

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

chat_sessions = {}

def get_or_create_session(user_id, firstname, skills, role, experience):
    # Reference the Firestore document for the user's session
    doc_ref = db.collection("sessions").document(user_id)
    doc = doc_ref.get()

    if doc.exists:
        session = doc.to_dict()
        # Update session details if they've changed
        session.update({
            "firstname": firstname,
            "skills": skills,
            "role": role,
            "experience": experience
        })
        # Update session data back to Firestore
        doc_ref.set(session, merge=True)
        logger.info(f"Updated Firestore session for user {user_id}")
    else:
        # Create a new session document
        session_id = str(uuid.uuid4())
        session = {
            "id": session_id,
            "user_id": user_id,
            "firstname": firstname,
            "skills": skills,
            "role": role,
            "experience": experience,
            "history": []
        }
        session["id"] = session_id  # 🔥 Ensure "id" field is written
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
        
        # Save the updated session back to Firestore
        doc_ref = db.collection("sessions").document(user_id)
        doc_ref.update({
            "history": chat_history,
            })

        # dispatch TTS
        #generate_tts_task.delay(answer, user_id, session["id"])

        return JSONResponse({
            "response": answer,
            "session_id": session["id"],
            "full_history": chat_history
        })

    except GoogleAPIError as e:
        logger.error(f"Google API Error: {e}", exc_info=True)
        raise HTTPException(status_code=502, detail="Error communicating with AI service.")
    except Exception as e:
        logger.error(f"Talk error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error.")

from datetime import timedelta

# @app.get("/get_audio/{user_id}")
# async def get_audio(user_id: str):
#     doc_ref = db.collection("sessions").document(user_id)
#     doc = doc_ref.get()

#     if not doc.exists:
#         raise HTTPException(status_code=404, detail="Session not found")

#     session_data = doc.to_dict()
#     if not session_data.get("last_audio_id"):
#         raise HTTPException(status_code=404, detail="Audio not ready yet")


#     audio_id = session_data["last_audio_id"]
#     blob = storage_client.bucket(bucket_name).blob(f"tts_audio/response_{audio_id}.mp3")
#     signed_url = blob.generate_signed_url(
#         version="v4",
#         expiration=timedelta(hours=1),
#         method="GET"
#     )
#     return JSONResponse({"audio_url": signed_url})

@app.post("/clear_chat/{user_id}")
async def clear_chat(user_id: str):
    # Fetch session from Firestore
    doc_ref = db.collection("sessions").document(user_id)
    doc = doc_ref.get()
    
    if doc.exists:
        session = doc.to_dict()
        session["history"] = []  # Clear the chat history
        doc_ref.set(session)  # Save the cleared history back to Firestore
        return {"message": "Chat history cleared"}
    else:
        raise HTTPException(status_code=404, detail="Session not found")

@app.post("/final_report")
async def final_report(user_id: str = Form(...)):
    # Fetch the session from Firestore
    doc_ref = db.collection("sessions").document(user_id)
    doc = doc_ref.get()
    
    if not doc.exists:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = doc.to_dict()
    chat_history = session.get("history", [])

    # Create prompt for the report generation
    prompt = (
f"""
You are a technical interviewer evaluating a candidate .

Based on the conversation history provided, write a **detailed and structured final evaluation report**. Your tone should be **professional, fair, and constructive**, suitable for use by a hiring manager.

Follow this format strictly:

---

**Final Evaluation of Candidate - [Role]**

**Overall Impression:**
Summarize the candidate's overall performance, focusing on their strengths, communication, and readiness for the role.

**Detailed Analysis by Question:**
For each major interview question or topic discussed, provide:
- A one-line topic title (e.g., “Challenging Project”)
- A short paragraph (2–3 sentences) evaluating their response
- A rating in the format: **(Marks: X/10)**

Use your judgment to estimate questions based on the dialogue — include 8–12 topics total.

**Strengths:**
- List 3–5 key strengths based on the conversation.

**Weaknesses:**
- List 3–5 key areas where the candidate can improve, focusing on clarity, depth, or approach.

**Suitability for the Role:**
State whether the candidate is suitable for the role. Use cautious, professional language (e.g., “suitable with guidance”, “strong potential”, or “may need further development”).

**Recommendations:**
Give practical advice the candidate should follow to grow professionally and succeed in this role.

---

Important Instructions:
- Base your evaluation only on what the candidate said — do not assume or fabricate skills.
- Be neutral in tone: supportive but honest.
- Do not include or mention date, time, or interview length.
- Format clearly with section headers as shown above.

Now, write the full evaluation report using the following interview conversation:
"""

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

