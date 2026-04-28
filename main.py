import io
import os
import json
from typing import Any, Dict, List, Optional

import httpx
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from openai import OpenAI
from pydub import AudioSegment

CHUNK_MS = 10 * 60 * 1000  # 10 minutes

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://meeting-frontend-ashy.vercel.app", "null", "file://"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory="static"), name="static")

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
    timeout=httpx.Timeout(300.0, connect=10.0),
)

class TranscriptIn(BaseModel):
    transcript: str

class ActionItem(BaseModel):
    task: str
    owner: Optional[str] = None
    due: Optional[str] = None  # keep strings like "Friday" for now

class Deadline(BaseModel):
    date: Optional[str] = None
    what: str
    owner: Optional[str] = None

class ProcessOut(BaseModel):
    transcript: Optional[str] = None
    summary: str
    action_items: List[ActionItem]
    deadlines: List[Deadline]
    decisions: List[str]


def _audio_fmt(filename: str, content_type: str) -> Optional[str]:
    if filename and "." in filename:
        ext = filename.rsplit(".", 1)[-1].lower()
        if ext in {"mp3", "wav", "ogg", "flac", "mp4", "m4a", "webm", "mpeg", "mpga", "aac"}:
            return ext
    ct_map = {
        "audio/mpeg": "mp3", "audio/mp3": "mp3",
        "audio/wav": "wav", "audio/x-wav": "wav", "audio/wave": "wav",
        "audio/ogg": "ogg", "audio/flac": "flac", "audio/x-flac": "flac",
        "audio/mp4": "mp4", "audio/m4a": "m4a", "audio/x-m4a": "m4a",
        "audio/webm": "webm", "video/webm": "webm", "video/mp4": "mp4",
        "audio/aac": "aac",
    }
    return ct_map.get(content_type)


def _transcribe_audio(filename: str, audio_bytes: bytes, content_type: str) -> str:
    fmt = _audio_fmt(filename, content_type)
    try:
        audio = AudioSegment.from_file(io.BytesIO(audio_bytes), format=fmt)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not decode audio: {e}")

    chunks = [audio[i : i + CHUNK_MS] for i in range(0, len(audio), CHUNK_MS)]

    parts = []
    for idx, chunk in enumerate(chunks):
        buf = io.BytesIO()
        chunk.export(buf, format="mp3")
        buf.seek(0)
        try:
            resp = client.audio.transcriptions.create(
                model="whisper-1",
                file=(f"chunk_{idx}.mp3", buf.read(), "audio/mpeg"),
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Whisper call failed on chunk {idx + 1}: {e}")
        parts.append(resp.text)

    return " ".join(parts)


def _extract_notes(transcript: str) -> dict:
    if not transcript:
        raise HTTPException(status_code=400, detail="Transcript is empty.")

    schema = {
        "name": "meeting_extract",
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "summary": {"type": "string"},
                "action_items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "task": {"type": "string"},
                            "owner": {"type": ["string", "null"]},
                            "due": {"type": ["string", "null"]},
                        },
                        "required": ["task", "owner", "due"],
                    },
                },
                "deadlines": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "date": {"type": ["string", "null"]},
                            "what": {"type": "string"},
                            "owner": {"type": ["string", "null"]},
                        },
                        "required": ["date", "what", "owner"],
                    },
                },
                "decisions": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
            "required": ["summary", "action_items", "deadlines", "decisions"],
        },
        "strict": True,
    }

    prompt = (
        "You are a meeting assistant. Extract information from the transcript.\n\n"
        "Return JSON that matches the schema.\n\n"
        "Rules:\n"
        "- summary: 1-3 sentences max.\n"
        "- action_items: only things someone must DO.\n"
        "- decisions: include explicit decisions (lines starting with 'Decision:' count).\n"
        "- deadlines: include any due dates/time-based commitments. If due day like 'Friday', keep it as the string.\n"
        "- If something is not mentioned, return an empty list for that field.\n\n"
        f"TRANSCRIPT:\n{transcript}"
    )

    try:
        resp = client.responses.create(
            model="gpt-4o-mini",
            input=prompt,
            text={
                "format": {
                    "type": "json_schema",
                    "name": schema["name"],
                    "schema": schema["schema"],
                    "strict": True,
                }
            },
        )
        raw = resp.output_text
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OpenAI call failed: {e}")

    try:
        data_out = json.loads(raw)
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail=f"Model returned non-JSON: {raw[:200]}")

    # Auto-create deadlines from action items that include due dates
    for item in data_out.get("action_items", []):
        due = item.get("due")
        if due:
            data_out.setdefault("deadlines", []).append({
                "date": due,
                "what": item.get("task"),
                "owner": item.get("owner"),
            })

    return data_out


@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    audio_bytes = await file.read()
    text = _transcribe_audio(file.filename, audio_bytes, file.content_type)
    return {"transcript": text}


@app.post("/process", response_model=ProcessOut)
def process(data: TranscriptIn):
    if not os.environ.get("OPENAI_API_KEY"):
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY is not set in your environment.")
    return _extract_notes(data.transcript.strip())


@app.post("/upload", response_model=ProcessOut)
async def upload(file: UploadFile = File(...)):
    if not os.environ.get("OPENAI_API_KEY"):
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY is not set in your environment.")
    audio_bytes = await file.read()
    transcript = _transcribe_audio(file.filename, audio_bytes, file.content_type)
    notes = _extract_notes(transcript)
    notes["transcript"] = transcript
    return notes


@app.get("/")
def index():
    return FileResponse("static/index.html")

@app.get("/health")
def health():
    return {"status": "ok"}
