"""Load documents (.txt/.md/.pdf) and split them into chunks with two strategies."""
import re
from dataclasses import dataclass, asdict
from pathlib import Path

from . import config


@dataclass
class Chunk:
    chunk_id: str
    doc: str
    section: str
    text: str

    def to_dict(self):
        return asdict(self)


def load_documents(raw_dir: Path = config.RAW_DIR) -> dict[str, str]:
    docs = {}
    for p in sorted(Path(raw_dir).glob("*")):
        if p.suffix.lower() in {".txt", ".md"}:
            docs[p.name] = p.read_text(encoding="utf-8")
        elif p.suffix.lower() == ".pdf":
            from pypdf import PdfReader

            reader = PdfReader(str(p))
            docs[p.name] = "\n".join((pg.extract_text() or "") for pg in reader.pages)
    return docs


def _split_sections(text: str) -> list[tuple[str, str]]:
    """Split on markdown-style headings ('# ', '## ') or numbered headings ('1. Title')."""
    sections, title, buf = [], "Intro", []
    heading = re.compile(r"^(#{1,4}\s+.+|\d+(\.\d+)*\.?\s+[A-Z][^\n]{2,80})$")
    for line in text.splitlines():
        if heading.match(line.strip()):
            if buf and "".join(buf).strip():
                sections.append((title, "\n".join(buf).strip()))
            title, buf = line.strip().lstrip("#").strip(), []
        else:
            buf.append(line)
    if buf and "".join(buf).strip():
        sections.append((title, "\n".join(buf).strip()))
    return sections


def chunk_fixed(doc: str, text: str, size: int, overlap: int) -> list[Chunk]:
    words = text.split()
    step = max(1, size - overlap)
    chunks = []
    for i, start in enumerate(range(0, len(words), step)):
        piece = words[start : start + size]
        if not piece:
            break
        chunks.append(Chunk(f"{doc}::fixed::{i}", doc, "", " ".join(piece)))
        if start + size >= len(words):
            break
    return chunks


def chunk_section(doc: str, text: str, max_words: int) -> list[Chunk]:
    """Heading-aware: keep a section together; split oversized ones on paragraphs."""
    chunks = []
    for title, body in _split_sections(text):
        paras = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
        buf, count, part = [], 0, 0
        for para in paras + [None]:
            w = len(para.split()) if para else 0
            if para is None or (buf and count + w > max_words):
                if buf:
                    # prefix heading so the chunk is self-describing for retrieval
                    txt = f"{title}\n" + "\n".join(buf)
                    chunks.append(Chunk(f"{doc}::section::{len(chunks)}", doc, title, txt))
                    part += 1
                buf, count = [], 0
            if para is not None:
                buf.append(para)
                count += w
    return chunks


def build_chunks(strategy: str = config.DEFAULT_STRATEGY, raw_dir: Path = config.RAW_DIR) -> list[Chunk]:
    params = config.CHUNK_STRATEGIES[strategy]
    out = []
    for name, text in load_documents(raw_dir).items():
        if strategy == "fixed":
            out += chunk_fixed(name, text, **params)
        else:
            out += chunk_section(name, text, **params)
    return out
