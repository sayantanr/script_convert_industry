import streamlit as st
import pytesseract
import zipfile, tempfile, os, re
from PIL import Image
from pdf2image import convert_from_path
from langdetect import detect
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate
import concurrent.futures

st.set_page_config(page_title="Mini OCR + Transliteration App", layout="centered")

st.title("📄 Mini OCR + Transliteration App")
st.caption("Handles ZIP uploads up to ~200 MB | Supports PDFs, Images, OCR, and Indic script conversion")

# -----------------------------
# OCR HELPERS
# -----------------------------
def ocr_image(image, lang="eng"):
    try:
        return pytesseract.image_to_string(image, lang=lang)
    except Exception as e:
        return f"[OCR ERROR] {e}"

def process_zip(zip_path, lang="eng"):
    extracted_texts = []
    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(tmpdir)
        image_files = sorted(
            [os.path.join(tmpdir, f) for f in os.listdir(tmpdir)
             if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        )
        if not image_files:
            return "[ERROR] No image files found in ZIP."

        progress = st.progress(0)
        step = 1 / len(image_files)

        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = {executor.submit(ocr_image, Image.open(img), lang): img for img in image_files}
            done = 0
            for future in concurrent.futures.as_completed(futures):
                fname = os.path.basename(futures[future])
                try:
                    text = future.result()
                    extracted_texts.append(f"\n\n--- PAGE: {fname} ---\n{text}")
                except Exception as e:
                    extracted_texts.append(f"[ERROR in {fname}]: {e}")
                done += 1
                progress.progress(min(done * step, 1.0))
    return "\n".join(extracted_texts)

# -----------------------------
# STREAMLIT UI
# -----------------------------
uploaded = st.file_uploader("Upload a TXT, PDF, Image, or ZIP (up to 200 MB)", type=["zip","pdf","png","jpg","jpeg","txt"])

ocr_lang = st.selectbox("OCR Language", ["eng","hin","ben"], index=0)
src_script = st.selectbox("Source Script", list(sanscript.SCHEMES.keys()), index=72)
tgt_script = st.selectbox("Target Script", list(sanscript.SCHEMES.keys()), index=5)

if uploaded:
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(uploaded.read())
        tmp.flush()

        if uploaded.name.endswith(".zip"):
            text_data = process_zip(tmp.name, lang=ocr_lang)
        elif uploaded.name.endswith(".pdf"):
            pages = convert_from_path(tmp.name)
            text_data = "\n".join([ocr_image(p, lang=ocr_lang) for p in pages])
        elif uploaded.name.lower().endswith((".jpg",".jpeg",".png")):
            text_data = ocr_image(Image.open(tmp.name), lang=ocr_lang)
        elif uploaded.name.endswith(".txt"):
            text_data = tmp.read().decode("utf-8", errors="ignore")
        else:
            st.error("Unsupported file type.")
            st.stop()

    st.subheader("🧾 Extracted Text Preview:")
    st.text_area("OCR Output", text_data[:4000], height=250)

    # Transliteration
    st.subheader("🔡 Transliteration Result:")
    translit = transliterate(text_data, src_script, tgt_script)
    st.text_area("Converted", translit[:4000], height=250)

    st.download_button("💾 Download Transliteration", translit, file_name="output.txt")

st.markdown("---")
st.caption("Developed by Sayantan Roy — OCR + Transliteration mini app (v1.0)")
