"""
src/extraction/text_extractor.py
---------------------------------
Unified text extraction from native documents and scanned images.

Supported inputs:
  - PDF  (native text layer via pdfplumber, per-page OCR fallback)
  - DOCX (python-docx)
  - Image / scanned PDF (Tesseract OCR)
  - Plain string (pass-through)

Returns a dict with keys:
  text        : str  — extracted text
  source      : str  — "native_pdf" | "mixed_pdf" | "native_docx" | "ocr" | "plain"
  pages       : int  — number of pages (PDF) or 0 for other types
  ocr_pages   : int  — number of pages that required OCR (0 for fully native docs)
  ocr_quality : str  — "good" | "low" | "empty" | "n/a"
                       "low"   = extracted text is very short (< 10 chars/page avg)
                       "empty" = extraction produced no text at all
                       "good"  = extraction produced reasonable text

GUIDE FEEDBACK FIXES (point D — OCR validation):
  1. Per-page detection: each PDF page is checked individually. Pages with no
     native text fall back to OCR rather than the whole document.
  2. Empty output check: if a page yields < 10 characters after stripping, it
     is flagged as potentially failed OCR.
  3. Quality flag in return dict: callers can check result["ocr_quality"] to
     decide whether to use the extracted text or skip the document.
  4. Page order is preserved — pages are joined in sequence with a page
     boundary marker so multi-column and ordered content is traceable.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Union

logger = logging.getLogger(__name__)

# Minimum characters per page to consider native extraction successful
_MIN_CHARS_PER_PAGE = 10


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_text(input_data: Union[str, Path], ocr_lang: str = "eng") -> dict:
    """
    Extract text from a document file or plain string.

    Parameters
    ----------
    input_data : str or Path
        File path to a PDF / DOCX / image, or a plain text string.
    ocr_lang   : str
        Tesseract language code (default: "eng").

    Returns
    -------
    dict with keys: text, source, pages, ocr_pages, ocr_quality
    """
    if isinstance(input_data, str) and not _looks_like_path(input_data):
        return {"text": input_data, "source": "plain", "pages": 0, "ocr_pages": 0, "ocr_quality": "n/a"}

    path = Path(input_data)
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        return _extract_pdf(path, ocr_lang)
    elif suffix in (".docx", ".doc"):
        return _extract_docx(path)
    elif suffix in (".png", ".jpg", ".jpeg", ".tiff", ".bmp"):
        return _extract_image_ocr(path, ocr_lang)
    else:
        try:
            text = path.read_text(encoding="utf-8")
            return {"text": text, "source": "plain", "pages": 0, "ocr_pages": 0, "ocr_quality": "n/a"}
        except Exception as exc:
            raise ValueError(f"Unsupported file type or unreadable file: {path}") from exc


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _looks_like_path(s: str) -> bool:
    """Heuristic: does the string look like a file path?"""
    return len(s) < 512 and (os.sep in s or s.endswith((".pdf", ".docx", ".png", ".jpg")))


def _extract_pdf(path: Path, ocr_lang: str) -> dict:
    """
    Per-page PDF extraction with individual OCR fallback.

    Each page is checked separately:
      - If a page has native text (>= _MIN_CHARS_PER_PAGE chars), it is used.
      - If a page has no usable native text, OCR is applied to that page only.

    This is more accurate than a whole-document fallback because many PDFs
    are "mixed" — some pages have text layers, some are scanned.
    The source is set to "mixed_pdf" when both methods were used.
    """
    try:
        import pdfplumber
    except ImportError as exc:
        raise ImportError(
            "pdfplumber is required for PDF extraction. Run: pip install pdfplumber"
        ) from exc

    page_texts = []
    ocr_page_indices = []

    with pdfplumber.open(path) as pdf:
        num_pages = len(pdf.pages)
        for i, page in enumerate(pdf.pages):
            native_text = (page.extract_text() or "").strip()
            if len(native_text) >= _MIN_CHARS_PER_PAGE:
                page_texts.append(native_text)
            else:
                logger.debug(
                    "Page %d/%d: native text too short (%d chars) — using OCR.",
                    i + 1, num_pages, len(native_text),
                )
                ocr_text = _ocr_single_pdf_page(path, i, ocr_lang)
                page_texts.append(ocr_text)
                ocr_page_indices.append(i)

    text = "\n\n--- page break ---\n\n".join(page_texts).strip()
    ocr_pages = len(ocr_page_indices)

    if ocr_pages == 0:
        source = "native_pdf"
    elif ocr_pages == num_pages:
        source = "ocr"
    else:
        source = "mixed_pdf"

    quality = _assess_quality(text, num_pages)
    if quality in ("low", "empty"):
        logger.warning(
            "PDF extraction quality '%s' for %s. "
            "OCR on %d/%d pages. Total chars: %d. "
            "Consider checking the document or using a different OCR engine.",
            quality, path.name, ocr_pages, num_pages, len(text),
        )
    else:
        logger.debug(
            "PDF extraction: source=%s, pages=%d, ocr_pages=%d, chars=%d, quality=%s",
            source, num_pages, ocr_pages, len(text), quality,
        )

    return {
        "text": text,
        "source": source,
        "pages": num_pages,
        "ocr_pages": ocr_pages,
        "ocr_quality": quality,
    }


def _ocr_single_pdf_page(path: Path, page_index: int, ocr_lang: str) -> str:
    """Render a single PDF page to an image and OCR it."""
    try:
        import fitz
        import pytesseract
        from PIL import Image
        import io
    except ImportError as exc:
        raise ImportError(
            "PyMuPDF and pytesseract are required for OCR. "
            "Run: pip install pymupdf pytesseract"
        ) from exc

    doc = fitz.open(str(path))
    page = doc[page_index]
    pix = page.get_pixmap(dpi=200)
    img = Image.open(io.BytesIO(pix.tobytes("png")))
    text = pytesseract.image_to_string(img, lang=ocr_lang)
    doc.close()
    return text.strip()


def _extract_docx(path: Path) -> dict:
    """Extract text from a DOCX file."""
    try:
        from docx import Document
    except ImportError as exc:
        raise ImportError("python-docx is required. Run: pip install python-docx") from exc

    doc = Document(str(path))
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    text = "\n".join(paragraphs)
    quality = _assess_quality(text, num_pages=max(len(paragraphs) // 30, 1))
    logger.debug("DOCX extraction: %d paragraphs, quality=%s", len(paragraphs), quality)
    return {"text": text, "source": "native_docx", "pages": 0, "ocr_pages": 0, "ocr_quality": quality}


def _extract_image_ocr(path: Path, ocr_lang: str) -> dict:
    """OCR a standalone image file."""
    try:
        import pytesseract
        from PIL import Image
    except ImportError as exc:
        raise ImportError(
            "pytesseract and Pillow are required for image OCR. "
            "Run: pip install pytesseract Pillow"
        ) from exc

    img = Image.open(str(path))
    text = pytesseract.image_to_string(img, lang=ocr_lang).strip()
    quality = _assess_quality(text, num_pages=1)
    if quality in ("low", "empty"):
        logger.warning(
            "Image OCR quality '%s' for %s (%d chars). "
            "Check image resolution and orientation.",
            quality, path.name, len(text),
        )
    logger.debug("Image OCR: %d chars, quality=%s", len(text), quality)
    return {"text": text, "source": "ocr", "pages": 1, "ocr_pages": 1, "ocr_quality": quality}


# ---------------------------------------------------------------------------
# Quality assessment
# ---------------------------------------------------------------------------

def _assess_quality(text: str, num_pages: int) -> str:
    """
    Assess extraction quality based on character count per page.

    Returns:
      "empty" — no text extracted at all
      "low"   — very short text (likely failed OCR or blank document)
      "good"  — reasonable amount of text
    """
    if not text or not text.strip():
        return "empty"
    chars_per_page = len(text) / max(num_pages, 1)
    if chars_per_page < _MIN_CHARS_PER_PAGE:
        return "low"
    return "good"


# ---------------------------------------------------------------------------
# Preprocessing utilities
# ---------------------------------------------------------------------------

def clean_text(text: str, *, max_chars: int | None = None) -> str:
    """
    Light cleaning: normalise whitespace, remove form-feed characters.
    Optionally truncate to max_chars.
    """
    import re
    text = re.sub(r"\f", "\n", text)
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.strip()
    if max_chars:
        text = text[:max_chars]
    return text


def chunk_text(text: str, max_tokens: int = 400, overlap_tokens: int = 50) -> list[str]:
    """
    Split text into overlapping word-level chunks so that each chunk fits
    within the model's context window.

    This is a simple word-count heuristic. A proper tokenizer-aware chunker
    is implemented in src/data/preprocessor.py.
    """
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + max_tokens, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start += max_tokens - overlap_tokens
    return chunks
