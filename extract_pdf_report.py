from pathlib import Path
from pypdf import PdfReader

pdf_path = Path(r"C:\Users\Aravind KJ\Downloads\TRUEVOICE - DEEPFAKE AUDIO DETECTION SYSTEM USING MEL-SPECTROGRAMS AND TRANSFER LEARNING (1).pdf")
out_path = Path(r"C:\Users\Aravind KJ\Desktop\deepfake 7\truevoice_pdf_extract.txt")

if not pdf_path.exists():
    print(f"ERROR: file not found: {pdf_path}")
    raise SystemExit(1)

reader = PdfReader(str(pdf_path))
chunks = []
non_empty_pages = 0
for i, page in enumerate(reader.pages, start=1):
    text = page.extract_text() or ""
    text = text.strip()
    if text:
        non_empty_pages += 1
    chunks.append(f"\n===== PAGE {i} =====\n{text}\n")

out_path.write_text("\n".join(chunks), encoding="utf-8", errors="ignore")

print(f"PAGES_TOTAL={len(reader.pages)}")
print(f"PAGES_WITH_TEXT={non_empty_pages}")
print(f"OUTPUT={out_path}")
