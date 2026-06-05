import os
import json
import uuid
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from openai import OpenAI

app = FastAPI()
templates = Jinja2Templates(directory="templates")

client = OpenAI(
    api_key=os.getenv("VLLM_API_KEY", "token"),
    base_url=os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1"),
)

MODEL = os.getenv("MODEL", "meta-llama/Llama-3.1-8B-Instruct")

# In-memory session store: {session_id: {"goal": str, "system_prompt": str, "history": [...]}}
sessions: dict = {}

ANALYZE_SYSTEM_PROMPT = """
You are an expert conversational analyst. Analyze the user's message in the context of a conversion conversation.

Return ONLY a valid JSON object:
{
  "intent": "ready_to_buy" | "interested" | "curious" | "objecting" | "stalling" | "not_interested" | "unknown",
  "tone": "positive" | "enthusiastic" | "neutral" | "skeptical" | "frustrated" | "negative",
  "pace": "brief" | "engaged" | "verbose",
  "meaning": "concise summary of what the user is actually communicating",
  "objections": ["specific objections or blockers, empty if none"],
  "buying_likelihood": 0.0 to 1.0,
  "notes": "any other signals worth noting"
}

No explanation outside the JSON.
""".strip()


# --- Models ---

class Message(BaseModel):
    role: str
    text: str

class AnalyzeRequest(BaseModel):
    user_input: str
    goal: str
    history: list[Message] = []

class StartRequest(BaseModel):
    goal: str
    system_prompt: str

class ChatRequest(BaseModel):
    session_id: str
    message: str


# --- UI ---

@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# --- Conversation API ---

@app.post("/api/start")
def start(req: StartRequest):
    session_id = str(uuid.uuid4())
    sessions[session_id] = {
        "goal": req.goal,
        "system_prompt": req.system_prompt,
        "history": [],
    }
    return {"session_id": session_id}


@app.post("/api/chat")
def chat(req: ChatRequest):
    session = sessions.get(req.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    session["history"].append({"role": "user", "content": req.message})

    messages = [{"role": "system", "content": session["system_prompt"]}] + session["history"]

    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0.7,
    )

    reply = response.choices[0].message.content
    session["history"].append({"role": "assistant", "content": reply})

    return {"response": reply}


@app.post("/api/reset")
def reset(req: dict):
    session_id = req.get("session_id")
    if session_id and session_id in sessions:
        sessions[session_id]["history"] = []
    return {"status": "reset"}


# --- Analyze endpoint (called by main Django app) ---

@app.post("/analyze")
def analyze(req: AnalyzeRequest):
    messages = [
        {"role": "assistant" if m.role == "agent" else "user", "content": m.text}
        for m in req.history
    ] + [{"role": "user", "content": req.user_input}]

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "system", "content": ANALYZE_SYSTEM_PROMPT}] + messages,
        temperature=0.2,
    )

    raw = response.choices[0].message.content
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start, end = raw.find("{"), raw.rfind("}") + 1
        if start == -1 or end <= start:
            raise HTTPException(status_code=502, detail=f"Invalid JSON from model: {raw}")
        return json.loads(raw[start:end])


@app.get("/health")
def health():
    return {"status": "ok"}
