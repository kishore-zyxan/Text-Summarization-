from PyPDF2 import PdfReader
import docx
import io

def read_file(file_content: bytes, filename: str) -> str:
    if filename.endswith(".pdf"):
        reader = PdfReader(io.BytesIO(file_content))
        return "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
    elif filename.endswith(".docx"):
        doc = docx.Document(io.BytesIO(file_content))
        return "\n".join([para.text for para in doc.paragraphs])
    elif filename.endswith(".txt"):
        return file_content.decode("utf-8")
    else:
        raise ValueError("Unsupported file format")
