import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from extractor import extract_text
from summarizer import summarize_large_doc
import logging
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI()

@app.post("/summarize")
async def summarize_document(file: UploadFile = File(...)):
    # Read file content
    start_total = time.time()
    content = await file.read()
    if len(content) > 5 * 1024 * 1024:  # Limit to 5MB
        logger.error(f"File {file.filename} exceeds 5MB limit")
        raise HTTPException(status_code=400, detail="File size exceeds 5MB limit")

    # Get file extension
    ext = os.path.splitext(file.filename)[-1].lower()
    logger.info(f"Processing file: {file.filename} (type: {ext})")

    try:
        # Extract text
        start_extract = time.time()
        text = extract_text(ext, content)
        extract_duration = time.time() - start_extract
        logger.info(f"Text extraction completed in {extract_duration:.2f} seconds")

        # Generate summary
        start_summarize = time.time()
        summary = summarize_large_doc(text)
        summarize_duration = time.time() - start_summarize
        logger.info(f"Summary generation completed in {summarize_duration:.2f} seconds")

        total_duration = time.time() - start_total
        logger.info(f"Total processing time for {file.filename}: {total_duration:.2f} seconds")

        return {"summary": summary}
    except Exception as e:
        total_duration = time.time() - start_total
        logger.error(f"Processing failed for {file.filename} after {total_duration:.2f} seconds: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")