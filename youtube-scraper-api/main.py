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
import markdown
from weasyprint import HTML

# --- CONFIGURATION ---
logging.basicConfig(level=logging.INFO)
app = FastAPI()
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

# --- HELPER FUNCTIONS ---
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
    *Instructions:*
    1.  *Extract Key Information:* Focus on key concepts, technical details, and actionable insights.
    2.  *Ignore Noise:* Skip repetitions, fillers, and transcription errors.
    3.  *Structure:* Use Markdown headings (##), bullet points (), and bold text (*).
    4.  *NO META-COMMENTARY:* Your final output must ONLY be the markdown notes.
    *Transcript Excerpt:* --- {chunk} --- *Detailed Markdown Notes:*
    """

def call_llm(prompt: str):
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {GROQ_API_KEY}"}
    data = {"model": "llama3-70b-8192", "messages": [{"role": "user", "content": prompt}]}
    
    try:
        response = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=data, timeout=120)
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"]
    except Exception as e:
        logging.error(f"Error calling LLM: {e}")
        return f"\n\n---\nError processing chunk: {e}\n---\n"

def scrape_and_generate_notes(video_url: str):
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
        
        for i, chunk in enumerate(chunks, start=1):
            logging.info(f"Processing chunk {i}/{len(chunks)}...")
            notes = call_llm(build_prompt(chunk))
            combined_notes.append(notes + "\n\n---\n\n")
            time.sleep(1)
        return video_title, "".join(combined_notes)

    except Exception as e:
        logging.error(f"Critical error in scrape_and_generate_notes: {e}")
        return "Error", f"A critical error occurred: {getattr(e, 'msg', str(e))}"
    finally:
        if os.path.exists(temp_vtt):
            os.remove(temp_vtt)

# --- API ENDPOINT ---
@app.post("/generate-notes-pdf")
async def generate_pdf_endpoint(request: Request):
    data = await request.json()
    video_url = data.get("url")
    if not video_url: return {"error": "No URL provided."}, 400

    title, markdown_notes = scrape_and_generate_notes(video_url)
    if title == "Error": return {"error": markdown_notes}, 500
    
    safe_filename = re.sub(r'[\\/*?:"<>|]', "", title) + ".pdf"
    
    # Convert Markdown to HTML, then HTML to PDF in memory
    html_text = markdown.markdown(markdown_notes)
    pdf_buffer = BytesIO()
    HTML(string=html_text).write_pdf(pdf_buffer)
    pdf_buffer.seek(0)

    return StreamingResponse(
        pdf_buffer,
        media_type='application/pdf',
        headers={'Content-Disposition': f'attachment; filename="{safe_filename}"'}
    )
