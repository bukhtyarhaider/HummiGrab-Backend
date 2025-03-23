import os
import re
from typing import Dict
import uuid
import threading
import logging
from flask import Flask, request, send_file, jsonify, render_template
from flask_cors import CORS
from werkzeug.utils import secure_filename
import yt_dlp
from dotenv import load_dotenv
from openai import OpenAI, OpenAIError
import time
import whisper
import backoff
import datetime

# New imports for database caching
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker

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
whisper_model = whisper.load_model("base")  # 'base' for speed; consider 'large' for improved accuracy

# Database setup (SQLite via SQLAlchemy)
DATABASE_URL = os.getenv('DATABASE_URL', 'sqlite:///video_records.db')
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

class VideoRecord(Base):
    __tablename__ = 'video_records'
    id = Column(Integer, primary_key=True, index=True)
    video_id = Column(String, unique=True, index=True)  
    url = Column(String)
    title = Column(String)
    transcript = Column(Text)
    summary = Column(Text)
    filename = Column(String, nullable=True)
    video_format = Column(String, nullable=True)   
    file_size = Column(Integer, nullable=True)       
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)


Base.metadata.create_all(bind=engine)

# Configuration
app.config['DOWNLOAD_FOLDER'] = os.getenv('DOWNLOAD_FOLDER', os.path.join(os.getcwd(), 'downloads'))
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB limit for request size
os.makedirs(app.config['DOWNLOAD_FOLDER'], exist_ok=True)

# In-memory download tracking (each download task is stored here)
download_tasks = {}

@app.after_request
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = 'http://localhost:5173'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    return response

@app.route('/')
def index():
    return jsonify({"message": "Backend is running, React frontend should handle routing."})

# Updated /get_info endpoint returns video_id
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

    video_id = info.get('id')  # Extract video ID from info
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
        'video_id': video_id,  # Return video id
        'title': title,
        'thumbnail': thumbnail,
        'duration': duration,
        'video_formats': video_formats
    })

@backoff.on_exception(
    backoff.expo,
    OpenAIError, 
    max_tries=5,
    giveup=lambda e: getattr(e.response, 'status_code', None) != 429
)
def generate_openai_summary(transcript: str) -> Dict[str, str]:
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
            temperature=0.7
        )
        summary_content = response.choices[0].message.content.strip()
        if not summary_content.startswith("# "):
            summary_content = "# Summary\n\n" + summary_content
        return {
            "summary": summary_content,
            "transcript": transcript
        }
    except AttributeError as e:
        raise OpenAIError(f"Unexpected response format from OpenAI API: {str(e)}")
    
@app.route('/generate_transcript', methods=['POST'])
def generate_transcript():
    data = request.get_json() or {}
    url = data.get('url')
    if not url:
        logger.error("No URL provided for transcript generation")
        return jsonify({'error': 'URL must be provided'}), 400

    # Extract video ID from the URL using a regex
    pattern = r'(?:youtube\.com\/(?:watch\?(?:.*&)?v=|embed\/|shorts\/)|youtu\.be\/)([A-Za-z0-9_-]{11})'
    match = re.search(pattern, url)
    if not match:
        logger.error("Invalid YouTube URL")
        return jsonify({'error': 'Invalid YouTube URL'}), 400
    video_id = match.group(1)
    logger.info(f"Extracted video_id: {video_id} from URL: {url}")

    # Check if a transcript already exists in the database for this video
    session = SessionLocal()
    video_record = session.query(VideoRecord).filter(VideoRecord.video_id == video_id).first()
    if video_record and video_record.transcript:
        transcript = video_record.transcript
        session.close()
        return jsonify({'transcript': transcript, 'video_id': video_id})

    # If no transcript exists, download the lowest-quality audio temporarily
    temp_id = uuid.uuid4().hex
    temp_file_template = os.path.join(app.config['DOWNLOAD_FOLDER'], f"temp_{temp_id}.%(ext)s")
    ydl_opts = {
        'outtmpl': temp_file_template,
        'noplaylist': True,
        'format': 'worstaudio',
        'quiet': True,
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            temp_file = ydl.prepare_filename(info)
            logger.info(f"Temporary audio downloaded: {temp_file}")
    except Exception as e:
        logger.error(f"Audio download failed: {str(e)}")
        session.close()
        return jsonify({'error': 'Audio download failed: ' + str(e)}), 500

    # Transcribe the downloaded audio using Whisper
    logger.info(f"Transcribing audio for video_id {video_id}")
    try:
        result = whisper_model.transcribe(temp_file)
    except Exception as e:
        logger.error(f"Transcription failed: {str(e)}")
        session.close()
        return jsonify({'error': 'Transcription failed: ' + str(e)}), 500
    transcript = result["text"]
    logger.info(f"Transcription completed for video_id {video_id}")

    # Store the transcript in the database
    if video_record is None:
        video_record = VideoRecord(
            video_id=video_id,
            title=info.get('title', ''),
            url=url,
            transcript=transcript
        )
        session.add(video_record)
    else:
        video_record.transcript = transcript
    session.commit()
    session.close()

    # Clean up the temporary audio file
    try:
        os.remove(temp_file)
        logger.info(f"Temporary file {temp_file} removed.")
    except Exception as e:
        logger.warning(f"Failed to remove temporary file {temp_file}: {str(e)}")

    return jsonify({'transcript': transcript, 'video_id': video_id})

@app.route('/generate_summary', methods=['POST'])
def generate_summary():
    data = request.get_json() or {}
    video_id = data.get('video_id')
    
    logger.info(f"Received summary request with data: {data}")
    
    session = SessionLocal()
    video_record = session.query(VideoRecord).filter(VideoRecord.video_id == video_id).first()
    
    # Require transcript to be present first
    if video_record is None or not video_record.transcript:
        session.close()
        logger.error(f"No avaiable Transcript for video_id: {video_id}: Transcript not generated. Please generate transcript first.")
        return jsonify({'error': 'Transcript not generated. Please generate transcript first.'}), 400
    
    transcript = video_record.transcript
    if video_record.summary:
        summary_text = video_record.summary
        session.close()
        return jsonify({
            'summary': summary_text,
            'video_id': video_id,
        })
    
    logger.info("Generating summary with OpenAI")
    try:
        response = generate_openai_summary(transcript)
        summary_text = response['summary']
        video_record.summary = summary_text
        session.commit()
    except OpenAIError as e:
        if getattr(e.response, 'status_code', None) == 429:
            logger.error("OpenAI quota exceeded or rate limit hit after retries")
            session.close()
            return jsonify({'error': 'OpenAI quota exceeded. Please check your plan or try again later.'}), 429
        session.close()
        raise

    session.close()
    return jsonify({
        'summary': summary_text,
        'video_id': video_id,
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

            session = SessionLocal()
            video_id = download_tasks[download_id].get('video_id')
            video_record = session.query(VideoRecord).filter(VideoRecord.video_id == video_id).first()
            if video_record:
                video_record.filename = filename
                video_record.video_format = download_tasks[download_id].get('video_format')
                if os.path.exists(filename):
                    video_record.file_size = os.path.getsize(filename)
                session.commit()
            session.close()
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
    video_format = data.get('video_format')  
    video_id = data.get('video_id')  
    title = data.get('title')        
    if not url or not video_format_id or not video_id:
        return jsonify({'error': 'URL, video_format_id, and video_id are required'}), 400

    download_id = uuid.uuid4().hex
    download_tasks[download_id] = {
        'url': url,
        'video_id': video_id,
        'title': title,
        'video_format': video_format,
        'progress': 0,
        'status': 'downloading',
        'filename': None,
        'cancelled': False
    }

    thread = threading.Thread(target=download_video_task, args=(download_id, url, video_format_id))
    thread.start()
    logger.info(f"Started download task {download_id} for video_id {video_id} and URL {url}")
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

# ============== Admin Dashboard Routes ==============

@app.route('/admin/dashboard')
def admin_dashboard():
    session = SessionLocal()
    records = session.query(VideoRecord).all()
    session.close()
    return render_template('admin_dashboard.html', records=records)

@app.route('/admin/delete_record/<int:record_id>', methods=['POST'])
def admin_delete_record(record_id):
    session = SessionLocal()
    record = session.query(VideoRecord).filter(VideoRecord.id == record_id).first()
    if not record:
        session.close()
        return jsonify({'error': 'Record not found'}), 404
    session.delete(record)
    session.commit()
    session.close()
    return jsonify({'message': 'Record deleted successfully'})

@app.route('/admin/delete_file/<int:record_id>', methods=['POST'])
def admin_delete_file(record_id):
    session = SessionLocal()
    record = session.query(VideoRecord).filter(VideoRecord.id == record_id).first()
    if not record:
        session.close()
        return jsonify({'error': 'Record not found'}), 404
    if record.filename and os.path.exists(record.filename):
        try:
            os.remove(record.filename)
        except Exception as e:
            session.close()
            return jsonify({'error': 'Error deleting file: ' + str(e)}), 500
        record.filename = None
        record.file_size = None  
        session.commit()
        session.close()
        return jsonify({'message': 'Cached file deleted successfully'})
    else:
        session.close()
        return jsonify({'error': 'No cached file found'}), 404


# ====================================================

if __name__ == "__main__":
    # Use waitress or gunicorn for production instead of Flask's dev server
    from waitress import serve
    port = int(os.getenv('PORT', 5000))
    logger.info(f"Starting server on port {port}")
    serve(app, host='0.0.0.0', port=port)
