from fastapi import APIRouter, UploadFile, File, Form, HTTPException
import shutil, tempfile, os
import pdfplumber
from docx import Document
from bs4 import BeautifulSoup
from PIL import Image
import pytesseract

from app.services.cv_eval.evaluation_engine import evaluation_engine
from app.services.cv_eval.improvement_engine import Improvement


router = APIRouter()

# -----------------------
# Helpers
# -----------------------

def extract_text_from_pdf(file_path: str) -> str:
    """Extract text from PDF using pdfplumber with a 7-page limit."""
    text = ""
    try:
        with pdfplumber.open(file_path) as pdf:
            if len(pdf.pages) > 7:
                raise ValueError(f"Resume exceeds maximum allowed length of 7 pages (found {len(pdf.pages)} pages).")
            
            for page in pdf.pages:
                text += (page.extract_text() or "") + "\n"
    except Exception as e:
        if "maximum allowed length" in str(e):
            raise
        # Fallback or other errors
        raise RuntimeError(f"Failed to extract PDF: {str(e)}")
        
    return text.strip()


def extract_text_from_docx(file_path: str) -> str:
    """Extract text from DOCX using python-docx."""
    doc = Document(file_path)
    return "\n".join([para.text for para in doc.paragraphs])


def extract_text_from_doc(file_path: str) -> str:
    """
    Extract text from .doc files (old MS Word format).
    Uses antiword (available in Docker image) via subprocess.
    """
    import subprocess
    try:
        result = subprocess.run(["antiword", file_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        return result.stdout.decode("utf-8", errors="ignore")
    except Exception as e:
        raise RuntimeError(f"Failed to extract .doc file: {e}")


def extract_text_from_txt(file_path: str) -> str:
    """Extract plain text from .txt or .md files."""
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def extract_text_from_rtf(file_path: str) -> str:
    """
    Extract text from RTF files by stripping control words.
    (Simple fallback; avoids textract.)
    """
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        data = f.read()
    import re
    return re.sub(r"{\\.*?}", "", data)


def extract_text_from_html(file_path: str) -> str:
    """Extract visible text from HTML using BeautifulSoup."""
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        soup = BeautifulSoup(f, "html.parser")
        return soup.get_text(separator="\n")


def extract_text_from_odt(file_path: str) -> str:
    """
    Extract text from ODT using odfpy (if installed),
    or fallback to zip XML extraction.
    """
    try:
        from odf.opendocument import load
        from odf.text import P
        doc = load(file_path)
        return "\n".join([p.firstChild.data for p in doc.getElementsByType(P) if p.firstChild])
    except Exception:
        import zipfile, xml.etree.ElementTree as ET
        text = ""
        with zipfile.ZipFile(file_path) as z:
            with z.open("content.xml") as f:
                tree = ET.parse(f)
                for elem in tree.iter():
                    if elem.text:
                        text += elem.text + "\n"
        return text.strip()


def extract_text_from_image(file_path: str) -> str:
    """Extract text from images (JPG/PNG/TIFF) via Tesseract OCR."""
    try:
        image = Image.open(file_path)
        return pytesseract.image_to_string(image)
    except Exception as e:
        raise RuntimeError(f"Failed to extract text from image: {e}")


def extract_text(file_path: str) -> str:
    """Dispatch extractor based on file extension."""
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        return extract_text_from_pdf(file_path)
    elif ext == ".docx":
        return extract_text_from_docx(file_path)
    elif ext == ".doc":
        return extract_text_from_doc(file_path)
    elif ext == ".txt":
        return extract_text_from_txt(file_path)
    elif ext == ".rtf":
        return extract_text_from_rtf(file_path)
    elif ext == ".md":
        return extract_text_from_txt(file_path)
    elif ext in [".html", ".htm"]:
        return extract_text_from_html(file_path)
    elif ext == ".odt":
        return extract_text_from_odt(file_path)
    elif ext in [".png", ".jpg", ".jpeg", ".tif", ".tiff"]:
        return extract_text_from_image(file_path)
    else:
        raise ValueError(f"Unsupported file format: {ext}")


def save_and_extract(upload: UploadFile) -> str:
    """Temporarily save uploaded file, extract its text, and enforce optimizations."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(upload.filename)[1]) as tmp:
        shutil.copyfileobj(upload.file, tmp)
        tmp_path = tmp.name
    try:
        text = extract_text(tmp_path)
        
        # Optimization: Enforce reasonable character limit for processing (around 7 dense pages)
        # 1 page is roughly 3000-4000 chars. 7 pages = ~25,000 chars.
        MAX_CHARS = 30000 
        if len(text) > MAX_CHARS:
            # We don't necessarily want to fail here if it's just text, 
            # but for optimization, we could trim or warn.
            # However, the user said "except resume of 7 pages at max", 
            # so for non-PDFs where we can't count pages accurately, 30k chars is a safe limit.
            if not upload.filename.lower().endswith('.pdf'):
                raise ValueError(f"Extracted content exceeds the equivalent of 7 pages (~{MAX_CHARS} characters).")
            
    finally:
        os.remove(tmp_path)
    return text


# -----------------------
# Endpoints
# -----------------------

@router.post("/cv_evaluate")
async def upload_and_evaluate_cv(
    file: UploadFile = File(...),
    jd_text: str = Form("", description="Optional JD text"),
    jd_file: UploadFile = File(None)
):
    """Upload CV (PDF/DOC/DOCX) and optionally JD for evaluation."""
    try:
        cv_text = save_and_extract(file)

        if jd_text and jd_text.strip():
            return evaluation_engine.evaluate(cv_text, jd_text)

        if jd_file is not None:
            jd_extracted = save_and_extract(jd_file)
            return evaluation_engine.evaluate(cv_text, jd_extracted)

        return evaluation_engine.evaluate(cv_text)

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Evaluation failed: {str(e)}")


# -----------------------
# Improvement Endpoint
# -----------------------

improvement_engine = Improvement()


@router.post("/cv_improvement")
async def upload_and_improve_cv(
    file: UploadFile = File(...),
    jd_text: str = Form("", description="JD text (optional)"),
    jd_file: UploadFile = File(None)
):
    """Upload CV and optionally JD → generate improved resume, benchmark, and cover letter."""
    try:
        cv_text = save_and_extract(file)

        if jd_text and jd_text.strip():
            return improvement_engine.evaluate(cv_text, jd_text)

        if jd_file is not None:
            jd_extracted = save_and_extract(jd_file)
            return improvement_engine.evaluate(cv_text, jd_extracted)

        # Fallback to generic JD-less improvement
        return improvement_engine.evaluate(cv_text, "")

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Improvement failed: {str(e)}")
