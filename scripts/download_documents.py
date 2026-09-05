"""
scripts/download_documents.py

Downloads the publicly available, official documentation PDFs listed in
data/document_sources.json into data/documents/.

- Skips files that are already downloaded (no duplicate downloads).
- Prints progress as it goes.
- Validates that downloaded files are genuinely PDFs (checks the %PDF magic
  bytes), and removes + reports any file that fails validation.
- Handles network/HTTP failures gracefully and continues with the rest of
  the list rather than crashing.

NOTE ON THE DOCUMENT CORPUS
----------------------------
Not every tool in this project's domain currently publishes an official,
freely downloadable PDF (for example, current NumPy/Matplotlib/scikit-learn
docs are mostly HTML/ZIP only, or PDF generation was discontinued upstream).
Where that was the case, this project supplements the corpus with original,
plainly-written study notes authored specifically for this project (see
scripts/generate_sample_notes.py) instead of fabricating a URL or bypassing
any access restriction. Feel free to replace/extend data/documents/ with
your own legitimately-sourced PDFs at any time — the pipeline (chunking,
embeddings, ChromaDB) re-indexes whatever is placed in that folder.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCES_FILE = PROJECT_ROOT / "data" / "document_sources.json"
DOCUMENTS_DIR = PROJECT_ROOT / "data" / "documents"

PDF_MAGIC = b"%PDF-"
REQUEST_TIMEOUT = 60
CHUNK_SIZE = 8192


def load_sources() -> list[dict]:
    if not SOURCES_FILE.exists():
        print(f"No document sources file found at {SOURCES_FILE}. Nothing to download.")
        return []
    with open(SOURCES_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def is_valid_pdf(path: Path) -> bool:
    try:
        with open(path, "rb") as f:
            return f.read(len(PDF_MAGIC)) == PDF_MAGIC
    except OSError:
        return False


def download_one(entry: dict) -> bool:
    title = entry.get("title", entry.get("filename", "unknown"))
    url = entry["url"]
    filename = entry["filename"]
    dest = DOCUMENTS_DIR / filename

    if dest.exists() and is_valid_pdf(dest):
        print(f"[SKIP] {title} -> {filename} already downloaded.")
        return True

    print(f"[DOWNLOAD] {title}\n           {url}")
    try:
        with requests.get(url, stream=True, timeout=REQUEST_TIMEOUT, headers={"User-Agent": "DataMind-RAG-Assistant/1.0"}) as response:
            response.raise_for_status()
            total = int(response.headers.get("content-length", 0))
            downloaded = 0
            tmp_path = dest.with_suffix(dest.suffix + ".part")
            with open(tmp_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                    if not chunk:
                        continue
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        pct = downloaded / total * 100
                        print(f"\r           {downloaded/1024:.0f} KB / {total/1024:.0f} KB ({pct:.0f}%)", end="")
            print()
    except requests.RequestException as exc:
        print(f"[FAILED] Could not download {title}: {exc}")
        return False

    if not is_valid_pdf(tmp_path):
        print(f"[FAILED] Downloaded file for {title} does not look like a valid PDF. Removing it.")
        tmp_path.unlink(missing_ok=True)
        return False

    tmp_path.rename(dest)
    print(f"[OK] Saved {filename}")
    return True


def main() -> int:
    DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
    sources = load_sources()

    if not sources:
        return 0

    print(f"Found {len(sources)} document source(s) to process.\n")

    successes = 0
    failures = 0
    for entry in sources:
        ok = download_one(entry)
        successes += int(ok)
        failures += int(not ok)
        print()

    print("=" * 60)
    print(f"Download summary: {successes} succeeded, {failures} failed.")
    if failures:
        print(
            "Some downloads failed (this can happen with no internet access, "
            "or if an upstream URL changed). You can still run "
            "scripts/generate_sample_notes.py and/or manually place PDFs "
            "into data/documents/ before building the vector store."
        )
    print("=" * 60)
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
