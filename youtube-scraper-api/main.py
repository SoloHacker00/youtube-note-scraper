from fastapi import FastAPI
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
GROQ_API_KEY = os.environ.get('GROQ_API_KEY')

# --- Helper Functions ---
def chunk_text(text, word_limit=600):
    """Splits text into chunks of a specified word limit."""
    chunks = []
    words = text.split()
    current_chunk = []
    for word in words:
        current_chunk.append(word)
        if len(current_chunk) >= word_limit:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

def get_notes_for_chunk(chunk):
    """Sends a chunk to the Groq API with an advanced prompt and robust error handling."""
    
    prompt = f"""
    You are a professional note-taking assistant. Your task is to analyze the following excerpt from a video transcript and generate detailed, well-structured notes in Markdown format.
    **Instructions:**
    1.  **Extract Key Information:** Focus strictly on extracting key concepts, technical details, step-by-step instructions, and actionable advice.
    2.  **Ignore Noise:** The transcript may contain repetition, transcription errors, and nonsensical phrases. Ignore these and focus only on the coherent, meaningful content.
    3.  **Structure the Output:** Organize the notes into logical sections using Markdown headings (##), bullet points (*), and bold text (**).
    4.  **DO NOT HALLUCINATE:** Do not invent context or add information that is not present in the excerpt.
    5.  **NO META-COMMENTARY:** Your final output must ONLY be the markdown notes. Do NOT include your thought process or any XML-style tags like `<think>`.
    **Transcript Excerpt:**
    ---
    {chunk}
    ---
    **Detailed Markdown Notes:**
    """
    
    # List of models to try in order of preference
    models_to_try = ["qwen/qwen3-32b","llama3-70b-8192", "mixtral-8x7b-32768", "gemma-7b-it"]

    for model in models_to_try:
        for attempt in range(3): # Try each model up to 3 times
            try:
                logging.info(f"Attempting to generate notes with model: {model}")
                response = requests.post(
                    url="https://api.groq.com/openai/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {GROQ_API_KEY}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": model,
                        "messages": [{"role": "user", "content": prompt}]
                    }
                )
                
                # Check for model decommissioned error specifically
                if response.status_code == 400 and 'model_decommissioned' in response.text:
                    logging.warning(f"Model {model} is decommissioned. Trying next model.")
                    break # Break the retry loop for this model and go to the next one

                response.raise_for_status() # Raise an exception for other bad status codes
                
                result = response.json()
                return result['choices'][0]['message']['content']
            
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 401:
                    return "\n\n---\nCRITICAL ERROR: 401 Unauthorized. Your Groq API key is invalid. Check your docker-compose.yml file.\n---\n"
                elif e.response.status_code == 429:
                    wait_time = 15#0 * (attempt + 1) # Exponential backoff
                    logging.warning(f"Rate limit hit. Waiting {wait_time} seconds before retry {attempt + 2}/3...")
                    time.sleep(wait_time)
                    continue
                else:
                    logging.error(f"HTTP Error for model {model}: {e.response.text}")
                    break # Stop trying this model if it gives a persistent error other than rate limiting
            except (KeyError, IndexError, Exception) as e:
                logging.error(f"Error parsing response or other issue for model {model}: {e}")
                break # Stop trying this model

    return "\n\n---\nCRITICAL ERROR: All tested models failed. Please check your API key, network connection, and the Groq model list.\n---\n"

def scrape_and_process_video(video_url):
    # This function remains largely the same
    temp_filename_base = "temp_notes"
    temp_vtt_filename_full = f"{temp_filename_base}.en.vtt"
    try:
        ydl_opts = {
            'writesubtitles': True, 'writeautomaticsub': True, 'subtitleslangs': ['en'],
            'subtitlesformat': 'vtt', 'skip_download': True, 'quiet': True,
            'outtmpl': temp_filename_base
        }
        if os.path.exists('/app/cookies.txt'):
            ydl_opts['cookiefile'] = '/app/cookies.txt'
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(video_url, download=True)
            video_title = info.get('title', 'Untitled Video')
        with open(temp_vtt_filename_full, 'r', encoding='utf-8') as f:
            vtt_content = f.read()
        captions = webvtt.read_buffer(StringIO(vtt_content))
        full_transcript = " ".join(dict.fromkeys([c.text.strip().replace('\n', ' ') for c in captions]))
        if not full_transcript:
            return video_title, "# No transcript available."
        text_chunks = chunk_text(full_transcript)
        combined_notes = [f"# Detailed Notes for: {video_title}\n\n"]
        for chunk in text_chunks:
            chunk_notes = get_notes_for_chunk(chunk)
            combined_notes.append(chunk_notes)
            combined_notes.append("\n\n---\n\n")
            time.sleep(1) # Keep a small base delay between successful calls
        return video_title, "".join(combined_notes)
    except Exception as e:
        return "Error", f"A critical error occurred: {getattr(e, 'msg', str(e))}"
    finally:
        if os.path.exists(temp_vtt_filename_full):
            os.remove(temp_vtt_filename_full)

@app.post("/generate-notes")
async def generate_notes_endpoint(request: dict):
    # This function is unchanged
    video_url = request.get("url")
    if not video_url:
        return {"error": "URL not provided"}
    title, notes = scrape_and_process_video(video_url)
    return {"title": title, "notes": notes}