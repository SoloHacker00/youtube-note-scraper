from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
import yt_dlp
import webvtt
from io import StringIO, BytesIO
import os
import re
import requests
import json
import logging
import time
from markdown2pdf import convert_md_to_pdf

# --- CONFIGURATION ---
logging.basicConfig(level=logging.INFO)
app = FastAPI()

# Default API keys from Render Environment Variables
DEFAULT_GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
DEFAULT_OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
DEFAULT_GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

# --- HELPER FUNCTIONS ---
def chunk_text(text, word_limit=600):
    """Splits text into chunks by word count."""
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
    """Creates the detailed markdown note-taking prompt."""
    return f"""
    You are a professional note-taking assistant. Your task is to analyze the following excerpt from a video transcript and generate detailed, well-structured notes in Markdown format.
    **Instructions:**
    1.  **Extract Key Information:** Focus on key concepts, technical details, and actionable insights.
    2.  **Ignore Noise:** Skip repetitions, fillers, and transcription errors.
    3.  **Structure:** Use Markdown headings (##), bullet points (*), and bold text (**).
    4.  **NO META-COMMENTARY:** Your final output must ONLY be the markdown notes.
    **Transcript Excerpt:** --- {chunk} --- **Detailed Markdown Notes:**
    """

def call_llm(platform: str, api_key: str, model: str, prompt: str):
    """Sends the prompt to the selected LLM platform with error handling."""
    headers = {"Content-Type": "application/json"}
    data = {}
    url = ""
    effective_key = api_key

    # --- Platform-specific configurations ---
    if platform == "groq":
        url = "https://api.groq.com/openai/v1/chat/completions"
        effective_key = api_key or DEFAULT_GROQ_API_KEY
        headers["Authorization"] = f"Bearer {effective_key}"
        data = {"model": model or "llama3-70b-8192", "messages": [{"role": "user", "content": prompt}]}
    
    elif platform == "openai":
        url = "https://api.openai.com/v1/chat/completions"
        effective_key = api_key or DEFAULT_OPENAI_API_KEY
        headers["Authorization"] = f"Bearer {effective_key}"
        data = {"model": model or "gpt-4o-mini", "messages": [{"role": "user", "content": prompt}]}

    elif platform == "google_gemini":
        effective_key = api_key or DEFAULT_GOOGLE_API_KEY
        model_name = model or "gemini-1.5-flash-latest"
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={effective_key}"
        data = {"contents": [{"parts": [{"text": prompt}]}]}
    
    else:
        return f"\n\n---\nError: Unsupported platform '{platform}'.\n---\n"

    if not effective_key:
        return f"\n\n---\nError: API key for {platform} is missing.\n---\n"

    # --- API Request with Retry Logic ---
    for attempt in range(3):
        try:
            response = requests.post(url, headers=headers, json=data, timeout=120)
            response.raise_for_status()
            result = response.json()

            if platform in ["groq", "openai"]:
                return result["choices"][0]["message"]["content"]
            elif platform == "google_gemini":
                return result["candidates"][0]["content"]["parts"][0]["text"]
        
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                wait_time = 15
                logging.warning(f"Rate limit hit. Waiting {wait_time}s before retry {attempt + 2}/3...")
                time.sleep(wait_time)
                continue
            else:
                logging.error(f"HTTP Error for {platform}: {e.response.text}")
                return f"\n\n---\nError with {platform} API: {e.response.text}\n---\n"
        except Exception as e:
            logging.error(f"Unexpected error for {platform}: {e}")
            return f"\n\n---\nError parsing {platform} response: {e}\n---\n"
    
    return f"\n\n---\nCRITICAL ERROR: Failed to get a response from {platform} after multiple retries.\n---\n"

def scrape_and_generate_notes(video_url: str, custom_api: dict):
    """Orchestrates the entire process from scrape to note generation."""
    temp_filename = "temp_video"
    temp_vtt = f"{temp_filename}.en.vtt"
    
    try:
        ydl_opts = {"writesubtitles": True, "writeautomaticsub": True, "subtitleslangs": ["en"], "subtitlesformat": "vtt", "skip_download": True, "quiet": True, "outtmpl": temp_filename}
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(video_url, download=True)
            video_title = info.get("title", "Untitled Video")

        with open(temp_vtt, "r", encoding="utf-8") as f:
            captions = webvtt.read_buffer(StringIO(f.read()))
        transcript = " ".join(dict.fromkeys([c.text.strip().replace("\n", " ") for c in captions]))

        if not transcript: return video_title, "# No transcript available."

        chunks = chunk_text(transcript)
        combined_notes = [f"# Detailed Notes for: {video_title}\n\n"]
        
        platform = custom_api.get("platform", "groq")
        api_key = custom_api.get("key")
        model = custom_api.get("model")

        for i, chunk in enumerate(chunks, start=1):
            logging.info(f"Processing chunk {i}/{len(chunks)}...")
            prompt = build_prompt(chunk)
            notes = call_llm(platform, api_key, model, prompt)
            combined_notes.append(notes + "\n\n---\n\n")
            time.sleep(1)

        return video_title, "".join(combined_notes)

    except Exception as e:
        logging.error(f"Critical error in scrape_and_generate_notes: {e}")
        return "Error", f"A critical error occurred: {getattr(e, 'msg', str(e))}"
    finally:
        if os.path.exists(temp_vtt):
            os.remove(temp_vtt)

# --- API ENDPOINTS ---
@app.post("/generate-pdf-notes")
async def generate_pdf_endpoint(request: Request):
    """Endpoint for generating a downloadable PDF (for web UI)."""
    data = await request.json()
    video_url = data.get("url")
    custom_api = data.get("custom_api", {})
    if not video_url: return {"error": "No URL provided."}, 400

    title, markdown_notes = scrape_and_generate_notes(video_url, custom_api)

    if title == "Error": return {"error": markdown_notes}, 500
    
    safe_filename = re.sub(r'[\\/*?:"<>|]', "", title) + ".pdf"
    
    try:
        pdf_buffer = BytesIO()
        convert_md_to_pdf(markdown_notes, pdf_buffer)
        pdf_buffer.seek(0)

        return StreamingResponse(
            pdf_buffer,
            media_type='application/pdf',
            headers={'Content-Disposition': f'attachment; filename="{safe_filename}"'}
        )
    except Exception as e:
        logging.error(f"PDF Conversion failed: {e}")
        return {"error": f"Failed to convert notes to PDF. Error: {e}"}, 500

