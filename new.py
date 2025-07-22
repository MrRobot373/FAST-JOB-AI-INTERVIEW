import os
import asyncio
import logging
import uuid
import json
import time
from typing import Dict, List, Optional, Any
from contextlib import asynccontextmanager
from datetime import datetime

# FastAPI imports
from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# Google imports
from google import genai
from google.genai import types
from google.api_core.exceptions import GoogleAPIError
from google.cloud import firestore

# Environment and logging
from dotenv import load_dotenv

# ==================== CONFIGURATION ====================
load_dotenv()

# Environment variables with defaults
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "30"))
GOOGLE_APPLICATION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "gcs-key.json")

# Set Google credentials
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = GOOGLE_APPLICATION_CREDENTIALS

# Validate critical settings
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY is required in environment variables")

# Configure logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL.upper()),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ==================== GLOBAL CLIENTS ====================

# Firestore client
try:
    firestore_client = firestore.Client()
    logger.info("Firestore client initialized")
except Exception as e:
    logger.error(f"Failed to initialize Firestore: {e}")
    raise

# Gemini client - NO SEMAPHORE LIMITATION
try:
    gemini_client = genai.Client(api_key=GEMINI_API_KEY)
    logger.info("Gemini client initialized - NO CONCURRENT REQUEST LIMITS")
except Exception as e:
    logger.error(f"Failed to initialize Gemini client: {e}")
    raise

# ==================== UTILITY FUNCTIONS ====================

async def get_firestore_document(collection: str, document_id: str) -> Optional[Dict[str, Any]]:
    """Get document from Firestore asynchronously"""
    try:
        loop = asyncio.get_event_loop()
        doc_ref = firestore_client.collection(collection).document(document_id)
        doc = await loop.run_in_executor(None, doc_ref.get)
        return doc.to_dict() if doc.exists else None
    except Exception as e:
        logger.error(f"Error getting Firestore document {document_id}: {e}")
        return None

async def set_firestore_document(collection: str, document_id: str, data: Dict[str, Any], merge: bool = True) -> bool:
    """Set document in Firestore asynchronously"""
    try:
        loop = asyncio.get_event_loop()
        doc_ref = firestore_client.collection(collection).document(document_id)
        await loop.run_in_executor(None, doc_ref.set, data, merge)
        return True
    except Exception as e:
        logger.error(f"Error setting Firestore document {document_id}: {e}")
        return False

async def update_firestore_document(collection: str, document_id: str, data: Dict[str, Any]) -> bool:
    """Update document in Firestore asynchronously"""
    try:
        loop = asyncio.get_event_loop()
        doc_ref = firestore_client.collection(collection).document(document_id)
        await loop.run_in_executor(None, doc_ref.update, data)
        return True
    except Exception as e:
        logger.error(f"Error updating Firestore document {document_id}: {e}")
        return False

def sync_generate_content(contents: List[types.Content], config: types.GenerateContentConfig) -> str:
    """Synchronous content generation for thread pool"""
    response_text = ""
    try:
        for chunk in gemini_client.models.generate_content_stream(
            model="gemini-2.0-flash-lite",
            contents=contents,
            config=config
        ):
            response_text += chunk.text
        return response_text
    except Exception as e:
        logger.error(f"Error in sync content generation: {e}")
        raise

async def generate_ai_response(contents: List[types.Content], timeout: int = None) -> str:
    """Generate AI response - NO CONCURRENCY CONTROL"""
    try:
        loop = asyncio.get_event_loop()
        
        config = types.GenerateContentConfig(
            response_mime_type="text/plain",
            temperature=0.0,
            top_p=1.0
        )
        
        # Run in thread pool to avoid blocking - NO SEMAPHORE
        future = loop.run_in_executor(None, sync_generate_content, contents, config)
        
        # Apply timeout
        timeout_duration = timeout or REQUEST_TIMEOUT
        response_text = await asyncio.wait_for(future, timeout=timeout_duration)
        
        return response_text.strip() or "Sorry, I couldn't generate a response. Please try again."
        
    except asyncio.TimeoutError:
        logger.error("Gemini API request timed out")
        return "Sorry, the request timed out. Please try again."
    except GoogleAPIError as e:
        logger.error(f"Google API Error: {e}")
        return "Sorry, there was an issue with the AI service. Please try again."
    except Exception as e:
        logger.error(f"Unexpected error in Gemini service: {e}")
        return "Sorry, an unexpected error occurred. Please try again."

async def get_or_create_session(user_id: str, firstname: str, skills: str, role: str, experience: str) -> Dict[str, Any]:
    """Get existing session or create new one"""
    try:
        # Try to get existing session
        session_data = await get_firestore_document("sessions", user_id)
        
        if session_data:
            # Update session with new details
            session_data.update({
                "firstname": firstname,
                "skills": skills,
                "role": role,
                "experience": experience
            })
            
            # Save updated session
            await set_firestore_document("sessions", user_id, session_data, merge=True)
            logger.info(f"Updated session for user {user_id}")
        else:
            # Create new session
            session_id = str(uuid.uuid4())
            session_data = {
                "id": session_id,
                "user_id": user_id,
                "firstname": firstname,
                "skills": skills,
                "role": role,
                "experience": experience,
                "history": []
            }
            
            await set_firestore_document("sessions", user_id, session_data)
            logger.info(f"Created new session {session_id} for user {user_id}")
        
        return session_data
        
    except Exception as e:
        logger.error(f"Error in get_or_create_session for user {user_id}: {e}")
        raise

# ==================== FASTAPI APPLICATION ====================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    logger.info("Starting AI Interview Bot application - UNLIMITED CONCURRENT USERS")
    yield
    logger.info("Shutting down AI Interview Bot application")

# Initialize FastAPI app
app = FastAPI(
    title="AI Interview Bot - Unlimited Users",
    description="Scalable AI-powered interview bot with unlimited concurrent users",
    version="2.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Mount static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# System instruction template
SYSTEM_INSTRUCTION = """
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
- If the candidate doesn't know, move on without offering solutions.
- Do NOT say "How can I help you?" or include asterisks.
""".strip()

# ==================== API ENDPOINTS ====================

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Serve the main interview page"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "concurrent_limits": "UNLIMITED",
        "supports_users": "UNLIMITED"
    }

@app.post("/talk")
async def talk(
    user_id: str = Form(...),
    firstname: str = Form(...),
    skills: str = Form(...),
    role: str = Form(...),
    experience: str = Form(...),
    message: str = Form(...),
):
    """Handle interview conversation - UNLIMITED CONCURRENT USERS"""
    start_time = time.time()
    
    try:
        # Get or create session
        session = await get_or_create_session(user_id, firstname, skills, role, experience)
        chat_history = session.get("history", [])
        
        # Build contents for Gemini API
        contents = [
            types.Content(
                role="assistant",
                parts=[types.Part.from_text(
                    text=SYSTEM_INSTRUCTION.format(role=role, firstname=firstname)
                )]
            )
        ]
        
        # Add chat history
        for entry in chat_history:
            api_role = "user" if entry["role"] == "user" else "assistant"
            contents.append(types.Content(
                role=api_role,
                parts=[types.Part.from_text(text=entry["text"])]
            ))
        
        # Add current message
        contents.append(types.Content(
            role="user",
            parts=[types.Part.from_text(text=message)]
        ))
        
        # Generate AI response - NO LIMITS
        response_text = await generate_ai_response(contents)
        
        # Update chat history
        updated_history = chat_history + [
            {"role": "user", "text": message},
            {"role": "assistant", "text": response_text}
        ]
        
        # Save updated history
        await update_firestore_document("sessions", user_id, {"history": updated_history})
        
        processing_time = time.time() - start_time
        logger.info(f"Interview conversation processed for user {user_id} in {processing_time:.2f}s")
        
        return JSONResponse({
            "response": response_text,
            "session_id": session["id"],
            "full_history": updated_history
        })
        
    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(f"Error in talk endpoint for user {user_id} after {processing_time:.2f}s: {e}")
        raise HTTPException(status_code=500, detail="Internal server error occurred")

@app.post("/clear_chat/{user_id}")
async def clear_chat(user_id: str):
    """Clear chat history for a user"""
    try:
        success = await update_firestore_document("sessions", user_id, {"history": []})
        
        if success:
            logger.info(f"Chat history cleared for user {user_id}")
            return {"message": "Chat history cleared successfully"}
        else:
            raise HTTPException(status_code=500, detail="Failed to clear chat history")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error clearing chat history for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error occurred")

@app.post("/final_report")
async def final_report(user_id: str = Form(...)):
    """Generate final interview report"""
    start_time = time.time()
    
    try:
        # Get session data
        session_data = await get_firestore_document("sessions", user_id)
        
        if not session_data:
            raise HTTPException(status_code=404, detail="Session not found")
        
        chat_history = session_data.get("history", [])
        
        if not chat_history:
            raise HTTPException(status_code=400, detail="No interview data available for report generation")
        
        # Create report generation prompt
        prompt = f"""
You are a technical interviewer evaluating a candidate.

Based on the conversation history provided, write a **detailed and structured final evaluation report**. Your tone should be **professional, fair, and constructive**, suitable for use by a hiring manager.

Follow this format strictly:

---

**Final Evaluation of Candidate - {session_data.get('role', 'Unknown Role')}**

**Overall Impression:**
Summarize the candidate's overall performance, focusing on their strengths, communication, and readiness for the role.

**Detailed Analysis by Question:**
For each major interview question or topic discussed, provide:
- A one-line topic title (e.g., "Challenging Project")
- A short paragraph (2–3 sentences) evaluating their response
- A rating in the format: **(Marks: X/10)**

Use your judgment to estimate questions based on the dialogue — include 8–12 topics total.

**Strengths:**
- List 3–5 key strengths based on the conversation.

**Weaknesses:**
- List 3–5 key areas where the candidate can improve, focusing on clarity, depth, or approach.

**Suitability for the Role:**
State whether the candidate is suitable for the role. Use cautious, professional language (e.g., "suitable with guidance", "strong potential", or "may need further development").

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
        
        # Add chat history
        for entry in chat_history:
            prompt += f"{entry['role'].capitalize()}: {entry['text']}\n"
        
        # Generate report - NO LIMITS
        contents = [types.Content(
            role="user",
            parts=[types.Part.from_text(text=prompt)]
        )]
        
        report_text = await generate_ai_response(contents, timeout=60)
        
        processing_time = time.time() - start_time
        logger.info(f"Final report generated for user {user_id} in {processing_time:.2f}s")
        
        return JSONResponse({"report": report_text})
        
    except HTTPException:
        raise
    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(f"Error generating final report for user {user_id} after {processing_time:.2f}s: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate report")

# ==================== MAIN ENTRY POINT ====================

if __name__ == "__main__":
    import uvicorn
    
    # Development server
    uvicorn.run(
        "unlimited_users_app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Remove in production
        log_level=LOG_LEVEL.lower(),
        access_log=True
    )
