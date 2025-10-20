from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
# --- THIS IS THE NEW PART: Import CORS middleware ---
from fastapi.middleware.cors import CORSMiddleware
import yt_dlp
import webvtt
from io import StringIO
import os
import re
import requests
import json
import logging
import time

# --- Configuration ---
logging.basicConfig(level=logging.INFO)
app = FastAPI()

# --- THIS IS THE CRUCIAL FIX ---
# Add the CORS middleware to allow requests from any domain
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

GROQ_API_KEY = os.environ.get('GROQ_API_KEY')

# --- Helper Functions (chunk_text, build_prompt, etc.) ---
# The rest of the code remains exactly the same.
def chunk_text(text, word_limit=600):
    words = text.split()
    chunks, current_chunk = [], []
    for word in words:
        current_chunk.append(word)
        if len(current_chunk) >= word_limit:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

def build_prompt(chunk: str) -> str:
    return f"""
    You are a professional note-taking assistant. Your task is to analyze the following excerpt from a video transcript and generate detailed, well-structured notes in Markdown format.
    **Instructions:**
    1.  **Extract Key Information:** Focus strictly on extracting key concepts, technical details, and actionable insights.
    2.  **Ignore Noise:** Skip repetitions, fillers, and transcription errors.
    3.  **Structure:** Use Markdown headings (##), bullet points (*), and bold text (**).
    4.  **NO META-COMMENTARY:** Your final output must ONLY be the markdown notes.
    **Transcript Excerpt:** --- {chunk} --- **Detailed Markdown Notes:**
    """

def call_llm(prompt: str):
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {GROQ_API_KEY}"}
    data = {"model": "llama3-70b-8192", "messages": [{"role": "user", "content": prompt}]}
    
    for attempt in range(3):
        try:
            response = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=data, timeout=120)
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"]
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                logging.warning(f"Rate limit hit. Waiting 15s before retry {attempt + 2}/3...")
                time.sleep(15)
                continue
            else:
                logging.error(f"HTTP Error: {e.response.text}")
                return f"\n\n---\nError with Groq API: {e.response.text}\n---\n"
        except Exception as e:
            logging.error(f"Unexpected error in call_llm: {e}")
            return f"\n\n---\nError parsing Groq response: {e}\n---\n"
    
    return "\n\n---\nCRITICAL ERROR: Failed to get a response from Groq after multiple retries.\n---\n"

def scrape_and_stream_notes(video_url: str):
    # This function is unchanged
    temp_filename = "temp_video"
    temp_vtt = f"{temp_filename}.en.vtt"
    try:
        ydl_opts = {"writesubtitles": True, "writeautomaticsub": True, "subtitleslangs": ["en"], "subtitlesformat": "vtt", "skip_download": True, "quiet": True, "outtmpl": temp_filename}
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(video_url, download=True)
            video_title = info.get("title", "Untitled Video")

        yield f"data: {json.dumps({'title': video_title})}\n\n"
        time.sleep(0.01)

        with open(temp_vtt, "r", encoding="utf-8") as f:
            captions = webvtt.read_buffer(StringIO(f.read()))
        transcript = " ".join(dict.fromkeys([c.text.strip().replace("\n", " ") for c in captions]))

        if not transcript:
            yield f"data: {json.dumps({'chunk': '# No transcript available.'})}\n\n"
            return

        chunks = chunk_text(transcript)
        for i, chunk in enumerate(chunks, start=1):
            logging.info(f"Processing chunk {i}/{len(chunks)}...")
            notes_chunk = call_llm(build_prompt(chunk))
            yield f"data: {json.dumps({'chunk': notes_chunk})}\n\n"
            time.sleep(0.01)

    except Exception as e:
        logging.error(f"Critical error in streaming process: {e}")
        error_message = f"A critical error occurred: {getattr(e, 'msg', str(e))}"
        yield f"data: {json.dumps({'error': error_message})}\n\n"
    finally:
        if os.path.exists(temp_vtt):
            os.remove(temp_vtt)
        logging.info("Streaming finished.")

@app.post("/stream-notes")
async def stream_notes_endpoint(request: Request):
    # This function is unchanged
    data = await request.json()
    video_url = data.get("url")
    if not video_url: return {"error": "No URL provided."}, 400

    return StreamingResponse(scrape_and_stream_notes(video_url), media_type="text/event-stream")

