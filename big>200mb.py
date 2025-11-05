import streamlit as st
import pytesseract
import zipfile, tempfile, os, re, random
from PIL import Image
from pdf2image import convert_from_path
from langdetect import detect
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate
import concurrent.futures
import joblib
import openai

# ==============================
# CONFIGURATION
# ==============================
st.set_page_config(page_title="AI Transliteration & OCR App", layout="wide")
openai.api_key = os.getenv("OPENAI_API_KEY")

LANGS = ["eng", "hin", "ben", "san", "ara"]
CACHE_DIR = "cache"
os.makedirs(CACHE_DIR, exist_ok=True)

# ==============================
# OCR FUNCTIONS
# ==============================
def ocr_image(image, lang="eng"):
    """Perform OCR on a single image safely."""
    try:
        return pytesseract.image_to_string(image, lang=lang)
    except Exception as e:
        return f"[OCR ERROR] {e}"

def process_zip_file(zip_path, lang="eng", max_workers=4):
    """Extract and OCR all images from a ZIP file, sorted by filename."""
    extracted_texts = []
    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(tmpdir)

        files = os.listdir(tmpdir)
        image_files, pdf_files = [], []
        for f in files:
            fp = os.path.join(tmpdir, f)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".tif", ".tiff")):
                image_files.append(fp)
            elif f.lower().endswith(".pdf"):
                pdf_files.append(fp)

        # Convert PDFs to images
        for pdf in pdf_files:
            pages = convert_from_path(pdf)
            for i, page in enumerate(pages):
                img_path = os.path.join(tmpdir, f"{os.path.splitext(os.path.basename(pdf))[0]}_{i}.jpg")
                page.save(img_path, "JPEG")
                image_files.append(img_path)

        image_files = sorted(image_files, key=lambda x: re.sub(r'\D', '', x))

        if not image_files:
            return "[ERROR] No image or PDF pages found."

        st.info(f"📄 Found {len(image_files)} pages. Running OCR...")
        progress = st.progress(0)
        step = 1 / len(image_files)

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_file = {executor.submit(ocr_image, Image.open(img), lang): img for img in image_files}
            done = 0
            for future in concurrent.futures.as_completed(future_to_file):
                img_name = os.path.basename(future_to_file[future])
                try:
                    text = future.result()
                    extracted_texts.append(f"\n\n--- PAGE: {img_name} ---\n{text}")
                except Exception as e:
                    extracted_texts.append(f"\n[ERROR on {img_name}]: {e}")
                done += 1
                progress.progress(min(done * step, 1.0))

    return "\n".join(extracted_texts)

# ==============================
# LANGUAGE DETECTION
# ==============================
def detect_script(text):
    try:
        lang = detect(text)
        return lang
    except:
        return "unknown"

# ==============================
# TRANSLITERATION + AI ENHANCEMENT
# ==============================
def transliterate_text(text, src_script, tgt_script, use_ai=False):
    """Transliterate and optionally correct via GPT."""
    try:
        base_output = transliterate(text, src_script, tgt_script)
        if use_ai and openai.api_key:
            prompt = f"Improve the transliteration accuracy from {src_script} to {tgt_script}:\n\n{base_output}"
            response = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=[{"role": "system", "content": "You are a transliteration expert."},
                          {"role": "user", "content": prompt}],
                temperature=0.2,
            )
            return response.choices[0].message.content.strip()
        return base_output
    except Exception as e:
        return f"[TRANSLITERATION ERROR] {e}"

# ==============================
# ACCURACY SCORING
# ==============================
def back_conversion_score(original, transliterated, src, tgt):
    """Estimate transliteration fidelity via back-conversion."""
    try:
        reconverted = transliterate(transliterated, tgt, src)
        sample = random.sample(range(len(original)), min(500, len(original)))
        matches = sum(original[i] == reconverted[i] for i in sample)
        return round((matches / len(sample)) * 100, 2)
    except:
        return 0.0

# ==============================
# MAIN STREAMLIT UI
# ==============================
st.title("🧠 AI-Powered Multi-Script Transliteration + OCR App")
st.caption("Supports 80+ scripts | OCR | GPT-4o smart correction | Accuracy scoring | Multi-file ZIPs up to 1 GB")

st.sidebar.header("⚙️ Options")
use_ai = st.sidebar.checkbox("Use OpenAI GPT-4o for smart correction", value=True)
ocr_lang = st.sidebar.selectbox("OCR language", LANGS, index=0)
src_script = st.sidebar.selectbox("Source Script", list(sanscript.SCHEMES.keys()), index=72)
tgt_script = st.sidebar.selectbox("Target Script", list(sanscript.SCHEMES.keys()), index=5)

uploaded = st.file_uploader("📂 Upload text, image, PDF, or ZIP of images:", type=["txt", "pdf", "png", "jpg", "jpeg", "zip"])

if uploaded:
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(uploaded.read())
        tmp.flush()

        if uploaded.name.endswith(".zip"):
            extracted_text = process_zip_file(tmp.name, lang=ocr_lang)
        elif uploaded.name.endswith(".pdf"):
            pages = convert_from_path(tmp.name)
            extracted_text = "\n".join([ocr_image(p, lang=ocr_lang) for p in pages])
        elif uploaded.name.lower().endswith((".png", ".jpg", ".jpeg")):
            extracted_text = ocr_image(Image.open(tmp.name), lang=ocr_lang)
        elif uploaded.name.endswith(".txt"):
            extracted_text = tmp.read().decode("utf-8", errors="ignore")
        else:
            st.error("Unsupported file type.")
            st.stop()

    st.subheader("🧾 Extracted Text:")
    st.text_area("OCR Output:", extracted_text[:5000], height=250)

    st.subheader("🔡 Transliteration:")
    translit_output = transliterate_text(extracted_text, src_script, tgt_script, use_ai)
    st.text_area("Result:", translit_output[:5000], height=250)

    score = back_conversion_score(extracted_text, translit_output, src_script, tgt_script)
    st.metric("Estimated Transliteration Accuracy (%)", score)

    st.download_button("💾 Download Transliteration", translit_output, file_name="transliteration_output.txt")

st.markdown("---")
st.markdown("💡 *Developed by Sayantan Roy — AI-powered linguistic technology for multi-script understanding.*")
