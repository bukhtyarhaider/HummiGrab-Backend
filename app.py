import os
from typing import Dict
import uuid
import threading
import logging
from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import yt_dlp
from dotenv import load_dotenv
from openai import OpenAI, OpenAIError
import time
import whisper
import backoff

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('app.log'), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": os.getenv("ALLOWED_ORIGINS", "*")}})

# Initialize OpenAI and Whisper
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
whisper_model = whisper.load_model("base")  # Use 'base' for speed, 'large' for better accuracy

# Configuration
app.config['DOWNLOAD_FOLDER'] = os.getenv('DOWNLOAD_FOLDER', os.path.join(os.getcwd(), 'downloads'))
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB limit for request size
os.makedirs(app.config['DOWNLOAD_FOLDER'], exist_ok=True)

download_tasks = {}

@app.after_request
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = 'http://localhost:5173'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    return response

# Routes
@app.route('/')
def index():
    return jsonify({"message": "Backend is running, React frontend should handle routing."})

@backoff.on_exception(
    backoff.expo,
    OpenAIError, 
    max_tries=5,
    giveup=lambda e: getattr(e.response, 'status_code', None) != 429  # Safely handle cases where response might not exist
)
def generate_openai_summary(transcript: str) -> Dict[str, str]:
    """
    Generate a well-formatted Markdown summary from a video transcript using OpenAI's GPT-4o-mini model.
    
    Args:
        transcript (str): The video transcript to summarize.
        
    Returns:
        Dict[str, str]: A dictionary containing:
            - 'summary': A detailed, structured, and objective summary in well-formatted Markdown.
            - 'transcript': The original transcript (returned as-is for frontend display).
            
    Raises:
        OpenAIError: If the API call fails after max retries, excluding rate limit errors (429).
        
    The summary will be formatted with:
    - A main heading (# Summary)
    - Bullet points for key points, main arguments, and supporting evidence
    - Direct quotes from the transcript where relevant
    - Proper spacing and structure for readability
    """
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a research assistant that provides detailed, structured, and objective summaries of video transcripts. "
                        "Your summaries should be in well-formatted Markdown and include:\n"
                        "- A small intoduction'\n"
                        "- Bullet points for key points, main arguments, and supporting evidence\n"
                        "- followed by concise bullet points using '-'. Use bold (**text**) for emphasis\n"
                        "- Direct quotes from the transcript where relevant, to support the summary points\n"
                        "- Proper spacing and structure for readability\n"
                        "Ensure that the summary is comprehensive, accurate, and captures the essence of the video content without introducing bias, suitable for research purposes."
                    )
                },
                {
                    "role": "user",
                    "content": f"Please summarize this video transcript:\n\n{transcript}"
                }
            ],
            max_tokens=600,
            temperature=0.7  # Added for controlled creativity
        )
        
        # Extract the summary from the response
        summary_content = response.choices[0].message.content.strip()
        
        # Ensure the response starts with a heading if not already present
        if not summary_content.startswith("# "):
            summary_content = "# Summary\n\n" + summary_content
        
        return {
            "summary": summary_content,
            "transcript": transcript  # Return original transcript as-is
        }
    
    except AttributeError as e:
        # Handle cases where response structure is unexpected
        raise OpenAIError(f"Unexpected response format from OpenAI API: {str(e)}")

@app.route('/generate_summary', methods=['POST'])
def generate_summary():
    data = request.get_json() or {}
    download_id = data.get('download_id')
    
    logger.info(f"Received summary request with data: {data}")
    logger.info(f"download_id: {download_id}")
    logger.info(f"Current download_tasks: {download_tasks}")
    
    if not download_id or download_id not in download_tasks:
        logger.error(f"Invalid download_id: {download_id} not found in download_tasks")
        return jsonify({'error': 'Invalid download_id'}), 400
    
    task = download_tasks[download_id]
    logger.info(f"Task details: {task}")
    
    if task.get('status') != 'completed' or not task.get('filename'):
        logger.warning(f"Task not ready: status={task.get('status')}, filename={task.get('filename')}")
        return jsonify({'error': 'Video not downloaded yet'}), 400
    
    video_path = task['filename']
    
    try:
        # Generate transcript using Whisper
        logger.info(f"Transcribing video: {video_path}")
        result = whisper_model.transcribe(video_path)
        transcript = result["text"]
        logger.info(f"Transcription completed: {transcript[:100]}...")
        
        # Generate summary using OpenAI with retry logic
        logger.info("Generating summary with OpenAI")
        try:
            response = generate_openai_summary(transcript)
            summary = response['summary']  # Access summary from the dictionary
        except OpenAIError as e:  # Correct exception name
            if getattr(e.response, 'status_code', None) == 429:
                logger.error("OpenAI quota exceeded or rate limit hit after retries")
                return jsonify({'error': 'OpenAI quota exceeded. Please check your plan or try again later.'}), 429
            raise  # Re-raise other errors
        
        logger.info(f"Summary generated: {summary[:50]}...")
        
        # Store summary with download task
        download_tasks[download_id]['summary'] = summary
        
        return jsonify({
            'summary': summary,
            'download_id': download_id,
            'transcript': transcript,
        })
        
    except Exception as e:
        logger.error(f"Error generating summary for {download_id}: {str(e)}")
        return jsonify({'error': f'Failed to generate summary: {str(e)}'}), 500

@app.route('/get_info', methods=['POST'])
def get_info():
    data = request.get_json() or {}
    url = data.get('url')
    if not url:
        return jsonify({'error': 'No URL provided'}), 400

    ydl_opts = {
        'quiet': True,
        'skip_download': True,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
    except Exception as e:
        logger.error(f"Error fetching info for URL {url}: {str(e)}")
        return jsonify({'error': 'Failed to fetch video info'}), 500

    title = info.get('title')
    thumbnail = info.get('thumbnail')
    duration = info.get('duration')

    video_formats = []
    for f in info.get('formats', []):
        if f.get('vcodec') != 'none' and f.get('acodec') == 'none':
            filesize_bytes = f.get('filesize') or f.get('filesize_approx')
            size_str = f" (~{filesize_bytes / 1048576:.2f} MB)" if filesize_bytes else ""
            resolution = f.get('height')
            resolution = f"{resolution}p" if resolution else f.get('format_note', '')
            label = f"{resolution} - {f.get('ext').upper()}{size_str}"
            video_formats.append({
                'format_id': f.get('format_id'),
                'label': label
            })

    return jsonify({
        'title': title,
        'thumbnail': thumbnail,
        'duration': duration,
        'video_formats': video_formats
    })

def download_progress_hook(download_id):
    def hook(d):
        if download_id not in download_tasks or download_tasks[download_id].get('cancelled'):
            raise Exception("Download cancelled")
        if d['status'] == 'downloading':
            total = d.get('total_bytes') or d.get('total_bytes_estimate')
            downloaded = d.get('downloaded_bytes', 0)
            percent = int(downloaded / total * 100) if total else 0
            download_tasks[download_id]['progress'] = percent
        elif d['status'] == 'finished':
            download_tasks[download_id]['progress'] = 100
    return hook

def download_video_task(download_id, url, video_format_id):
    ydl_opts = {
        'outtmpl': os.path.join(app.config['DOWNLOAD_FOLDER'], '%(title)s.%(ext)s'),
        'noplaylist': True,
        'format': f"{video_format_id}+bestaudio",
        'merge_output_format': 'mp4',
        'progress_hooks': [download_progress_hook(download_id)]
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            filename = ydl.prepare_filename(info)
            if not filename.endswith('.mp4'):
                base, _ = os.path.splitext(filename)
                filename = base + '.mp4'
            download_tasks[download_id]['filename'] = filename
            download_tasks[download_id]['status'] = 'completed'
            logger.info(f"Download completed for {download_id}: {filename}")
    except Exception as e:
        download_tasks[download_id]['status'] = 'error'
        download_tasks[download_id]['error'] = str(e)
        logger.error(f"Download failed for {download_id}: {str(e)}")
    finally:
        if download_id in download_tasks and download_tasks[download_id].get('cancelled'):
            if 'filename' in download_tasks[download_id] and os.path.exists(download_tasks[download_id]['filename']):
                os.remove(download_tasks[download_id]['filename'])
            del download_tasks[download_id]

@app.route('/start_download', methods=['POST'])
def start_download():
    data = request.get_json() or {}
    url = data.get('url')
    video_format_id = data.get('video_format_id')
    if not url or not video_format_id:
        return jsonify({'error': 'URL and video_format_id are required'}), 400

    download_id = uuid.uuid4().hex
    download_tasks[download_id] = {
        'progress': 0,
        'status': 'downloading',
        'filename': None,
        'cancelled': False
    }

    thread = threading.Thread(target=download_video_task, args=(download_id, url, video_format_id))
    thread.start()
    logger.info(f"Started download task {download_id} for URL {url}")
    return jsonify({'download_id': download_id})

@app.route('/progress', methods=['GET'])
def progress():
    download_id = request.args.get('download_id')
    if not download_id or download_id not in download_tasks:
        return jsonify({'error': 'Invalid download_id'}), 404
    return jsonify(download_tasks[download_id])

@app.route('/cancel_download', methods=['POST'])
def cancel_download():
    data = request.get_json() or {}
    download_id = data.get('download_id')
    if not download_id or download_id not in download_tasks:
        return jsonify({'error': 'Invalid download_id'}), 404
    
    if download_tasks[download_id]['status'] != 'downloading':
        return jsonify({'error': 'No active download to cancel'}), 400
    
    download_tasks[download_id]['cancelled'] = True
    download_tasks[download_id]['status'] = 'error'
    download_tasks[download_id]['error'] = 'Download cancelled by user'
    logger.info(f"Download {download_id} cancelled by user")
    return jsonify({'message': 'Download cancellation requested'})

@app.route('/get_file', methods=['GET'])
def get_file():
    download_id = request.args.get('download_id')
    if not download_id or download_id not in download_tasks:
        return jsonify({'error': 'Invalid download_id'}), 400
    task = download_tasks[download_id]
    if task.get('status') != 'completed' or not task.get('filename'):
        return jsonify({'error': 'File not ready'}), 400
    
    filename = task['filename']
    return send_file(filename, as_attachment=True, download_name=secure_filename(os.path.basename(filename)))

def cleanup_downloads():
    for download_id, task in list(download_tasks.items()):
        if task.get('status') in ['completed', 'error'] and 'filename' in task:
            if os.path.exists(task['filename']) and (time.time() - os.path.getmtime(task['filename']) > 24 * 3600):
                os.remove(task['filename'])
                download_tasks[download_id] = {
                    'status': task['status'],
                    'summary': task.get('summary'),
                    'progress': task.get('progress', 100 if task['status'] == 'completed' else 0)
                }
                logger.info(f"Cleaned up file for {download_id}, keeping task data")
            elif not task.get('summary') and not os.path.exists(task['filename']):
                del download_tasks[download_id]
                logger.info(f"Removed empty task {download_id}")

if __name__ == "__main__":
    #Use waitress or gunicorn for production instead of Flask's dev server
    from waitress import serve
    port = int(os.getenv('PORT', 5000))
    logger.info(f"Starting server on port {port}")
    serve(app, host='0.0.0.0', port=port)
    # app.run(debug=True)