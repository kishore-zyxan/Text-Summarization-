import io
import hashlib
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import docx
import pandas as pd
from pypdf import PdfReader
from PIL import Image
import easyocr
from pdf2image import convert_from_bytes
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

reader = easyocr.Reader(['en'], gpu=False)
extraction_cache = {}

@lru_cache(maxsize=100)
def extract_text(ext: str, content_tuple: tuple) -> str:
    content = bytes(content_tuple)
    content_hash = hashlib.md5(content).hexdigest()
    if content_hash in extraction_cache:
        logger.info("Returning cached text for content hash")
        return extraction_cache[content_hash]

    if ext == ".pdf":
        try:
            logger.info("Attempting PDF text extraction with pypdf")
            reader = PdfReader(io.BytesIO(content))
            with ThreadPoolExecutor() as executor:
                text = "".join(executor.map(lambda p: p.extract_text() or "", reader.pages))
        except Exception as e:
            logger.warning(f"pypdf extraction failed: {str(e)}. Falling back to OCR")
            try:
                # Convert PDF to images (limit to first 10 pages for performance)
                images = convert_from_bytes(content, first_page=1, last_page=10)
                text = ""
                for i, image in enumerate(images):
                    logger.info(f"Processing page {i+1} with OCR")
                    image = image.convert("L").resize((300, 300))
                    page_text = " ".join(reader.readtext(image, detail=0))
                    text += page_text + "\n"
            except Exception as ocr_e:
                logger.error(f"OCR extraction failed: {str(ocr_e)}")
                raise ValueError(f"Failed to extract text from PDF: {str(ocr_e)}")

    elif ext == ".docx":
        logger.info("Extracting text from DOCX")
        doc = docx.Document(io.BytesIO(content))
        text = "\n".join(para.text for para in doc.paragraphs)

    elif ext in [".png", ".jpg", ".jpeg"]:
        logger.info("Extracting text from image with OCR")
        image = Image.open(io.BytesIO(content)).convert("L").resize((300, 300))
        text = " ".join(reader.readtext(image, detail=0))

    elif ext == ".csv":
        logger.info("Extracting text from CSV")
        df = pd.read_csv(io.BytesIO(content))
        text = df.to_string(index=False)

    elif ext == ".txt":
        logger.info("Extracting text from TXT")
        text = content.decode("utf-8")

    else:
        logger.error(f"Unsupported file type: {ext}")
        raise ValueError(f"Unsupported file type: {ext}")

    extraction_cache[content_hash] = text
    logger.info(f"Cached text for content hash, length: {len(text)} characters")
    return text