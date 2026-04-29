import glob
import json
import os
import shutil
import subprocess
import tempfile
from typing import Any, Dict, List, Optional

import httpx
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from openai import OpenAI

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
    due: str

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


def _transcribe_audio(filename: str, audio_bytes: bytes, content_type: str) -> str:
    suffix = ("." + filename.rsplit(".", 1)[-1].lower()) if filename and "." in filename else ""
    tmpdir = tempfile.mkdtemp()
    try:
        input_path = os.path.join(tmpdir, f"input{suffix}")
        with open(input_path, "wb") as f:
            f.write(audio_bytes)

        segment_pattern = os.path.join(tmpdir, "chunk_%03d.mp3")
        cmd = [
            "ffmpeg", "-y", "-i", input_path,
            "-f", "segment", "-segment_time", "600",
            "-vn", "-acodec", "libmp3lame", "-ab", "128k",
            segment_pattern,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise HTTPException(status_code=400, detail=f"ffmpeg failed: {result.stderr[-500:]}")

        segment_files = sorted(glob.glob(os.path.join(tmpdir, "chunk_*.mp3")))
        if not segment_files:
            raise HTTPException(status_code=400, detail="ffmpeg produced no output segments.")

        parts = []
        for idx, seg_path in enumerate(segment_files):
            with open(seg_path, "rb") as f:
                seg_bytes = f.read()
            try:
                resp = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=(os.path.basename(seg_path), seg_bytes, "audio/mpeg"),
                )
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Whisper call failed on chunk {idx + 1}: {e}")
            parts.append(resp.text)

        return " ".join(parts)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


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
                            "due": {"type": "string"},
                        },
                        "required": ["task", "due"],
                    },
                },
                "deadlines": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "date": {"type": "string"},
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
        "You are a precise meeting note-taker. Extract structured information from the transcript below.\n"
        "Follow every instruction exactly — the same transcript must always produce the same output.\n\n"
        "SUMMARY\n"
        "Write a structured summary covering every major topic discussed in the meeting.\n"
        "Use one short paragraph per topic. Do not skip or truncate any topic.\n\n"
        "ACTION ITEMS\n"
        "Extract every task, follow-up, or commitment mentioned by any speaker.\n"
        "Include implicit commitments (e.g. 'I'll handle that') as well as explicit ones.\n"
        "For each action item:\n"
        "  - task: the specific thing to be done, stated clearly\n"
        "  - due: the due date exactly as stated (e.g. 'Friday', 'end of quarter', 'March 15th'),\n"
        "         or 'Not specified' if no due date was mentioned — never null or empty\n\n"
        "DEADLINES\n"
        "List every date-bound commitment or milestone mentioned.\n"
        "  - what: what is due\n"
        "  - owner: who is responsible, or null if not mentioned\n"
        "  - date: the date exactly as stated, or 'Not specified' if no date was mentioned — never null or empty\n\n"
        "DECISIONS\n"
        "List every decision made, one per entry. Include decisions stated explicitly ('We decided...')\n"
        "and implicit ones (e.g. agreement to proceed with a plan). If none, return an empty array.\n\n"
        f"TRANSCRIPT:\n{transcript}"
    )

    try:
        resp = client.responses.create(
            model="gpt-4o-mini",
            input=prompt,
            temperature=0,
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

    # Ensure due/date fields are never null or empty (guards against model non-compliance)
    for item in data_out.get("action_items", []):
        if not item.get("due"):
            item["due"] = "Not specified"
    for dl in data_out.get("deadlines", []):
        if not dl.get("date"):
            dl["date"] = "Not specified"

    # Auto-create deadlines from action items that have a real due date
    for item in data_out.get("action_items", []):
        due = item.get("due")
        if due and due != "Not specified":
            data_out.setdefault("deadlines", []).append({
                "date": due,
                "what": item.get("task"),
                "owner": None,
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
