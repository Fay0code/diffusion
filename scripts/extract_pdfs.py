from pathlib import Path
import json
import pdfplumber

root = Path(__file__).resolve().parents[1]
source = root / "paper" / "负载"
out = root / "tmp" / "pdfs" / "extracted"
out.mkdir(parents=True, exist_ok=True)

manifest = []
for pdf_path in sorted(source.glob("*.pdf")):
    with pdfplumber.open(pdf_path) as pdf:
        pages = []
        for i, page in enumerate(pdf.pages, 1):
            text = page.extract_text(x_tolerance=2, y_tolerance=3) or ""
            pages.append(f"\n\n===== PAGE {i} =====\n{text}")
        text_path = out / f"{pdf_path.stem}.txt"
        text_path.write_text("".join(pages), encoding="utf-8")
        manifest.append({
            "file": str(pdf_path),
            "pages": len(pdf.pages),
            "metadata": pdf.metadata,
            "text_file": str(text_path),
            "characters": sum(len(x) for x in pages),
        })

(out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(manifest, ensure_ascii=False, indent=2))
