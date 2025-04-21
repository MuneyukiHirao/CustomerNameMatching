import logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(levelname)s %(message)s")
from fastapi import FastAPI, UploadFile, File, HTTPException, Body, BackgroundTasks, Query
from fastapi.responses import FileResponse, RedirectResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent))
from pipeline import init_db, process_upload
from db import SessionLocal
from models import MatchScore
from datetime import datetime
from uuid import uuid4
import io
import glob
import threading
import asyncio
import json

# Load environment variables
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)

app = FastAPI()

# In-memory job tracking
jobs: dict = {}

@app.get("/")
async def health_check():
    """Health check endpoint to verify server is running."""
    return {"status": "running"}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

## Run DB init in background on startup to prevent blocking
@app.on_event("startup")
def on_startup():
    # Synchronous DB initialization to ensure tables ready before serving
    logging.info("on_startup: running init_db synchronously")
    init_db()
    logging.info("on_startup: database initialization complete")

def _process_job(job_id: str, fileA_bytes: bytes, fileB_bytes: bytes, filenameA: str, filenameB: str):
    """Background task to run pipeline and store progress and result."""
    # define progress callback
    def progress(msg: str):
        jobs[job_id]["messages"].append(msg)
        jobs[job_id]["status"] = msg
    try:
        logging.info(f"Job {job_id}: starting process_upload")
        jobs[job_id]["status"] = "running"
        jobs[job_id]["messages"].append("Job started")
        # reconstruct file-like objects for pipeline
        class DummyUpload:
            def __init__(self, data: bytes, name: str):
                self.file = io.BytesIO(data)
                self.filename = name
        a = DummyUpload(fileA_bytes, filenameA)
        b = DummyUpload(fileB_bytes, filenameB)
        url = process_upload(a, b, progress_callback=progress)
        logging.info(f"Job {job_id}: process_upload completed, URL={url}")
        jobs[job_id]["status"] = "completed"
        jobs[job_id]["download_url"] = url
    except Exception as e:
        logging.error(f"Job {job_id} failed: {e}")
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)

@app.post("/upload")
async def upload(
    background_tasks: BackgroundTasks,
    fileA: UploadFile = File(...),
    fileB: UploadFile = File(...),
    debug: bool = Query(False, description="Run processing synchronously for debugging")
):
    """Trigger matching pipeline asynchronously, return job ID."""
    # read file bytes before request ends
    dataA = await fileA.read()
    dataB = await fileB.read()
    if debug:
        logging.info("Debug mode: running process_upload synchronously")
        # Wait for background DB initialization to complete
        try:
            from pipeline import table_init_done
            table_init_done.wait(timeout=30)
        except ImportError:
            pass
        class DummyUpload:
            def __init__(self, data: bytes, name: str):
                self.file = io.BytesIO(data)
                self.filename = name
        a = DummyUpload(dataA, fileA.filename)
        b = DummyUpload(dataB, fileB.filename)
        url = process_upload(a, b)
        return {"status": "completed", "download_url": url}
    # Test mode: run synchronously even when debug=False
    if os.getenv("TESTING"):
        job_id = str(uuid4())
        jobs[job_id] = {"status": "pending", "messages": [], "download_url": None, "error": None}
        class DummyUpload:
            def __init__(self, data: bytes, name: str):
                self.file = io.BytesIO(data)
                self.filename = name
        a = DummyUpload(dataA, fileA.filename)
        b = DummyUpload(dataB, fileB.filename)
        url = process_upload(a, b)
        jobs[job_id]["status"] = "completed"
        jobs[job_id]["download_url"] = url
        return {"job_id": job_id}
    # asynchronous background job
    job_id = str(uuid4())
    jobs[job_id] = {"status": "pending", "messages": [], "download_url": None, "error": None}
    background_tasks.add_task(_process_job, job_id, dataA, dataB, fileA.filename, fileB.filename)
    return {"job_id": job_id}

@app.get("/status/{job_id}")
async def get_status(job_id: str):
    """Get status and download URL for a job."""
    info = jobs.get(job_id)
    if not info:
        raise HTTPException(status_code=404, detail="Job not found")
    return info

@app.put("/review/{cand_id}")
async def review(
    cand_id: str,
    match: bool = Body(..., embed=True),
    user: str = Body(..., embed=True)
):
    """Update review status of a candidate match."""
    session = SessionLocal()
    score = session.get(MatchScore, cand_id)
    if not score:
        raise HTTPException(status_code=404, detail="Candidate not found")
    score.status = 'approved' if match else 'rejected'
    score.action_user = user
    score.action_time = datetime.utcnow()
    session.commit()
    return {"status": "saved"}

@app.get("/download")
async def download(job_id: str):
    """Redirect to blob SAS URL for completed job."""
    info = jobs.get(job_id)
    if not info:
        raise HTTPException(status_code=404, detail="Job not found")
    if info.get("status") != "completed":
        raise HTTPException(status_code=400, detail="Job not completed yet")
    url = info.get("download_url")
    if not url:
        raise HTTPException(status_code=404, detail="Download URL not available")
    return RedirectResponse(url)

@app.get("/status/stream/{job_id}")
async def stream_status(job_id: str):
    """SSE endpoint streaming job progress events."""
    async def event_generator():
        last = 0
        while True:
            job = jobs.get(job_id)
            if not job:
                yield f"event: error\ndata: {json.dumps({'error':'Job not found'})}\n\n"
                return
            msgs = job.get('messages', [])
            # send new messages
            for msg in msgs[last:]:
                data = {'step': msg}
                yield f"event: progress\ndata: {json.dumps(data)}\n\n"
            last = len(msgs)
            # end event on completion or failure
            if job.get('status') in ('completed', 'failed'):
                yield f"event: end\ndata: {json.dumps({'status': job.get('status'), 'download_url': job.get('download_url')})}\n\n"
                return
            await asyncio.sleep(1)
    return StreamingResponse(event_generator(), media_type='text/event-stream')

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
