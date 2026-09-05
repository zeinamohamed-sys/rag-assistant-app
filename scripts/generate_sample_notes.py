"""
scripts/generate_sample_notes.py

Renders the original, project-authored study notes in
scripts/sample_notes_content.py into real PDF files under data/documents/.

This exists to supplement the corpus for topics where no official,
freely-redistributable PDF is currently published (see the note at the top
of scripts/download_documents.py). All text here was written specifically
for this project — it is not copied from any external source.
"""

from __future__ import annotations

from pathlib import Path

from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer

from sample_notes_content import DOCUMENTS

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOCUMENTS_DIR = PROJECT_ROOT / "data" / "documents"


def build_pdf(filename: str, title: str, sections: list[tuple[str, list[str]]]) -> Path:
    dest = DOCUMENTS_DIR / filename
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "DataMindTitle", parent=styles["Title"], fontSize=20, spaceAfter=18
    )
    heading_style = ParagraphStyle(
        "DataMindHeading", parent=styles["Heading2"], spaceBefore=14, spaceAfter=8
    )
    body_style = ParagraphStyle(
        "DataMindBody", parent=styles["BodyText"], fontSize=11, leading=16, spaceAfter=10
    )

    doc = SimpleDocTemplate(
        str(dest),
        pagesize=LETTER,
        leftMargin=0.9 * inch,
        rightMargin=0.9 * inch,
        topMargin=0.9 * inch,
        bottomMargin=0.9 * inch,
        title=title,
    )

    story = [Paragraph(title, title_style), Spacer(1, 6)]
    story.append(
        Paragraph(
            "Original study notes prepared for the DataMind RAG Assistant project.",
            body_style,
        )
    )

    for heading, paragraphs in sections:
        story.append(Paragraph(heading, heading_style))
        for para in paragraphs:
            story.append(Paragraph(para, body_style))

    doc.build(story)
    return dest


def main() -> None:
    DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Generating {len(DOCUMENTS)} original study-note PDF(s) into {DOCUMENTS_DIR}...\n")
    for filename, spec in DOCUMENTS.items():
        dest = build_pdf(filename, spec["title"], spec["sections"])
        size_kb = dest.stat().st_size / 1024
        print(f"[OK] {filename} ({size_kb:.0f} KB) - {spec['title']}")
    print("\nDone.")


if __name__ == "__main__":
    main()
