from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from extractor import extract_text
from summarizer import summarize_text
import os

app = FastAPI()


@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    try:
        content = await file.read()
        ext = os.path.splitext(file.filename)[1].lower()

        # Extract text from uploaded file
        extracted_text = extract_text(ext, content)
        if not extracted_text.strip():
            return JSONResponse(status_code=400, content={"error": "No text found in file"})

        # Run map-reduce summarization
        summary = summarize_text(extracted_text)

        return {
            "extracted_text": extracted_text,
            "summary": summary
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
