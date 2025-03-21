# HummiGrab API

A lightweight Flask-based backend API that leverages [yt-dlp](https://github.com/yt-dlp/yt-dlp) to fetch video information and download videos. This application provides endpoints to retrieve video metadata, initiate downloads, monitor download progress, cancel downloads, and serve the downloaded file.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [API Endpoints](#api-endpoints)
- [Logging](#logging)
- [Production Deployment](#production-deployment)
- [License](#license)

## Features

- **Video Info Retrieval:** Fetch video metadata such as title, thumbnail, duration, and available formats.
- **Video Downloading:** Download videos using yt-dlp with progress tracking.
- **Download Management:** Check progress, cancel downloads, and retrieve the completed file.
- **Cross-Origin Support:** Configured CORS to allow requests from specified origins.
- **Logging:** Built-in logging for monitoring and troubleshooting.
- **Production Ready:** Uses [Waitress](https://docs.pylonsproject.org/projects/waitress/en/stable/) for serving the application in production.

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/your-repo-name.git
   cd your-repo-name
   ```

2. **Create and activate a virtual environment (optional but recommended):**

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

   > **Note:** Ensure the `requirements.txt` includes:
   > - Flask
   > - Flask-Cors
   > - python-dotenv
   > - yt-dlp
   > - waitress
   > - Werkzeug

## Configuration

Create a `.env` file in the project root to set your environment variables. Example:

```dotenv
ALLOWED_ORIGINS=http://localhost:5173
DOWNLOAD_FOLDER=./downloads
PORT=5000
```

- **ALLOWED_ORIGINS:** Specifies allowed CORS origins.
- **DOWNLOAD_FOLDER:** Directory where downloaded videos are saved.
- **PORT:** Port number on which the server will run.

## Usage

1. **Run the server:**

   For production, the code uses Waitress:

   ```bash
   python app.py
   ```

   This starts the server on the port defined in your `.env` file (default is `5000`).

2. **Development:**

   If you want to use Flask’s development server for testing, you can uncomment the `app.run(debug=True)` line at the end of `app.py` and run:

   ```bash
   python app.py
   ```

## API Endpoints

### Base URL
The API is accessible via `http://<your-server-host>:<port>`.

### Endpoints

- **`GET /`**  
  Returns a basic JSON message confirming that the backend is running.

  **Response:**
  ```json
  { "message": "Backend is running, React frontend should handle routing." }
  ```

- **`POST /get_info`**  
  Fetches metadata for a video.

  **Request Body:**
  ```json
  { "url": "https://www.youtube.com/watch?v=example" }
  ```

  **Response:**
  ```json
  {
    "title": "Video Title",
    "thumbnail": "Thumbnail URL",
    "duration": 300,
    "video_formats": [
      { "format_id": "18", "label": "360p - MP4 (~3.45 MB)" }
    ]
  }
  ```

- **`POST /start_download`**  
  Initiates the download of a video in a selected format.

  **Request Body:**
  ```json
  {
    "url": "https://www.youtube.com/watch?v=example",
    "video_format_id": "18"
  }
  ```

  **Response:**
  ```json
  { "download_id": "unique_download_identifier" }
  ```

- **`GET /progress`**  
  Retrieves the current progress of a download.

  **Query Parameter:**
  ```
  download_id=unique_download_identifier
  ```

  **Response:**
  ```json
  {
    "progress": 50,
    "status": "downloading",
    "filename": null,
    "cancelled": false
  }
  ```

- **`POST /cancel_download`**  
  Cancels an ongoing download.

  **Request Body:**
  ```json
  { "download_id": "unique_download_identifier" }
  ```

  **Response:**
  ```json
  { "message": "Download cancellation requested" }
  ```

- **`GET /get_file`**  
  Serves the downloaded video file for retrieval once the download is complete.

  **Query Parameter:**
  ```
  download_id=unique_download_identifier
  ```

  **Response:**  
  The file is sent as an attachment.

## Logging

Logs are written to both `app.log` and the console. They include timestamps, log levels, and detailed messages for download tasks and errors.

## Production Deployment

For production environments, it is recommended to use a WSGI server like [Waitress](https://docs.pylonsproject.org/projects/waitress/en/stable/) (already included) or [Gunicorn](https://gunicorn.org/). Make sure to properly configure environment variables and consider using a reverse proxy (e.g., Nginx) for better performance and security.

## License

This project is licensed under the [MIT License](LICENSE).