import fitz  # PyMuPDF
from PIL import Image
import io

def extract_text_multilang(
    pdf_path, 
    lang="eng",           # "eng", "ben", "hin", "urd", etc.
    ocr_langs=None
):
    """
    Extract text from a PDF using Tesseract for most languages, EasyOCR for Urdu.
    - pdf_path: path to PDF file
    - lang: string code (default "eng"), use "urd" for Urdu
    - ocr_langs: list of fallback languages
    Returns: string containing extracted text from all pages
    """
    if ocr_langs is None:
        ocr_langs = [lang]

    doc = fitz.open(pdf_path)
    all_text = []

    for page_num, page in enumerate(doc):
        text = page.get_text().strip()
        if text and lang != "urd":
            # If digital text and not Urdu, use as is:
            all_text.append(text)
            continue

        # Else: Render page as image and OCR
        pix = page.get_pixmap()
        img = Image.open(io.BytesIO(pix.tobytes()))

        if lang == "urd":
            try:
                import easyocr
                reader = easyocr.Reader(['ur'])
                img_path = f"temp_page_{page_num+1}.png"
                img.save(img_path)
                result = reader.readtext(img_path, detail=0, paragraph=True)
                ocr_text = "\n".join(result)
                all_text.append(ocr_text)
            except Exception as e:
                all_text.append(f"[EasyOCR Urdu extraction failed: {e}]")
        else:
            import pytesseract
            ocr_text = ""
            for olang in ocr_langs:
                ocr_text = pytesseract.image_to_string(img, lang=olang)
                if ocr_text.strip():
                    break
            all_text.append(ocr_text)
    return "\n".join(all_text)
