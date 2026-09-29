from __future__ import annotations

import argparse
import json
import re
import zipfile
from pathlib import Path
from xml.etree import ElementTree

from pypdf import PdfReader


def clean_text(value: str) -> str:
    return " ".join(str(value).split())


def text_chunks(value: str, size: int, overlap: int):
    value = clean_text(value)
    step = max(1, size - overlap)
    for start in range(0, len(value), step):
        piece = value[start:start + size]
        if len(piece.strip()) >= 120:
            yield piece.strip()


def pdf_text(path: Path, max_pages: int) -> str:
    reader = PdfReader(str(path))
    pages = []
    for page in reader.pages[:max(1, max_pages)]:
        try:
            pages.append(page.extract_text() or "")
        except Exception:
            continue
    return "\n".join(pages)


def docx_text(path: Path) -> str:
    namespace = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
    with zipfile.ZipFile(path) as archive:
        root = ElementTree.fromstring(archive.read("word/document.xml"))
    return "\n".join(node.text or "" for node in root.iter(namespace + "t"))


def records_for(path: Path, *, max_pages: int, chunk_size: int, chunk_overlap: int):
    if path.suffix.lower() == ".pdf":
        try:
            text = pdf_text(path, max_pages)
        except Exception:
            return
    elif path.suffix.lower() in {".md", ".txt"}:
        text = path.read_text(encoding="utf-8", errors="ignore")
    elif path.suffix.lower() == ".docx":
        text = docx_text(path)
    else:
        return
    for index, chunk in enumerate(text_chunks(text, chunk_size, chunk_overlap)):
        yield {
            "text": chunk,
            "meta": {"source": path.name, "page_block": index, "kind": "paper", "modality": "visual"},
        }


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest research papers into JSONL text streams")
    parser.add_argument("--source", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-pages", type=int, default=10)
    parser.add_argument("--chunk-size", type=int, default=720)
    parser.add_argument("--chunk-overlap", type=int, default=80)
    parser.add_argument("--max-chunks-per-file", type=int, default=3)
    args = parser.parse_args()

    source = Path(args.source)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    counts = {}
    files = sorted(p for p in source.rglob("*") if p.is_file() and p.suffix.lower() in {".pdf", ".md", ".txt", ".docx"})
    for path in files:
        group = path.parent.relative_to(source).as_posix().replace("\\", "_").replace("/", "_") or "root"
        out_path = output_dir / f"{group or 'root'}.jsonl"
        written = 0
        with out_path.open("a", encoding="utf-8") as handle:
            for record in records_for(path, max_pages=args.max_pages, chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap):
                if written >= max(0, args.max_chunks_per_file):
                    break
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                written += 1
        counts[str(path)] = written
    print(json.dumps({"files": len(files), "chunks": sum(counts.values()), "outputs": sorted(output_dir.glob('*.jsonl'))}, ensure_ascii=False, default=str, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
