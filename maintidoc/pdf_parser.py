"""Analyse d'un PDF : extraction du texte, des tableaux et des images.

- Texte  : LlamaIndex `PyMuPDFReader` (un Document LlamaIndex par page).
- Tableaux : détection PyMuPDF (`page.find_tables`) puis reconstruction ligne
  par ligne (cellules séparées par « | ») à partir de la position des mots.
- Images : images matricielles intégrées au PDF + figures vectorielles
  (schémas dessinés), repérées grâce à leur légende « Figure N » puis
  rendues en PNG.

Chaque élément garde ses métadonnées (document, page, type, légende) pour
pouvoir citer des sources vérifiables.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path

import fitz  # PyMuPDF
from llama_index.core import Document
from llama_index.readers.file import PyMuPDFReader

from .config import settings

FIGURE_CAPTION_RE = re.compile(r"^\s*(Figure|Fig\.)\s*\d+", re.IGNORECASE)
TABLE_CAPTION_RE = re.compile(r"^\s*(Table|Tableau)\s*\d+", re.IGNORECASE)

MIN_IMAGE_SIDE = 60   # en points PDF : ignore icônes, logos minuscules, puces
RENDER_ZOOM = 2.5     # résolution du rendu des figures (≈ 180 dpi)


@dataclass
class ExtractedTable:
    page: int
    markdown: str
    caption: str = ""


@dataclass
class ExtractedImage:
    page: int
    path: Path
    caption: str = ""
    bbox: tuple[float, float, float, float] = (0, 0, 0, 0)


@dataclass
class ParsedPDF:
    doc_id: str
    doc_name: str
    path: Path
    num_pages: int
    pages: list[Document] = field(default_factory=list)
    tables: list[ExtractedTable] = field(default_factory=list)
    images: list[ExtractedImage] = field(default_factory=list)


def file_doc_id(path: Path) -> str:
    """Identifiant stable du document (hash du contenu) : ré-indexer remplace l'ancien."""
    return hashlib.sha1(path.read_bytes()).hexdigest()[:16]


# ---------------------------------------------------------------- texte
def extract_text(path: Path, doc_id: str, doc_name: str) -> list[Document]:
    reader = PyMuPDFReader()
    docs = reader.load_data(file_path=path, metadata=True)
    pages = []
    for i, d in enumerate(docs, start=1):
        text = d.text.strip()
        if not text:
            continue
        pages.append(
            Document(
                text=text,
                metadata={"doc_id": doc_id, "doc_name": doc_name, "page": i, "type": "text"},
            )
        )
    return pages


# ------------------------------------------------------------ légendes
def _text_blocks(page: fitz.Page) -> list[tuple[fitz.Rect, str]]:
    return [
        (fitz.Rect(b[:4]), b[4].strip())
        for b in page.get_text("blocks")
        if b[6] == 0 and b[4].strip()
    ]


def _caption_blocks(page: fitz.Page, regex: re.Pattern) -> list[tuple[fitz.Rect, str]]:
    return [(r, " ".join(t.split())) for r, t in _text_blocks(page) if regex.match(t)]


def _nearest_caption(rect: fitz.Rect, captions: list[tuple[fitz.Rect, str]], max_gap: float = 60) -> str:
    """Légende la plus proche juste au-dessus ou au-dessous d'une zone."""
    best, best_gap = "", max_gap
    for crect, text in captions:
        gap = min(abs(crect.y0 - rect.y1), abs(rect.y0 - crect.y1))
        if gap < best_gap:
            best, best_gap = text, gap
    return best


# ------------------------------------------------------------ tableaux
def _page_may_have_table(page: fitz.Page, captions: list) -> bool:
    """`find_tables` est coûteux : on ne l'appelle que si une légende « Table N »
    est présente ou si la page contient beaucoup de traits horizontaux/verticaux."""
    if captions:
        return True
    segments = 0
    for d in page.get_drawings():
        for item in d["items"]:
            if item[0] == "l" or (item[0] == "re" and min(item[1].width, item[1].height) < 2):
                segments += 1
    return segments >= 6


def _table_rows_text(page: fitz.Page, rect: fitz.Rect, col_gap: float = 8) -> str:
    """Reconstruit le tableau ligne par ligne à partir de la position des mots.
    Les cellules d'une même ligne sont séparées par « | ». Plus robuste que la
    grille de `find_tables` pour les tableaux sans traits entre les lignes."""
    words = sorted(page.get_text("words", clip=rect), key=lambda w: (round(w[3]), w[0]))
    lines: list[list] = []
    for w in words:
        if lines and abs(lines[-1][0] - w[3]) < 3:
            lines[-1][1].append(w)
        else:
            lines.append([w[3], [w]])
    out = []
    for _, ws in lines:
        ws.sort(key=lambda w: w[0])
        s = ws[0][4]
        for a, b in zip(ws, ws[1:]):
            s += (" | " if b[0] - a[2] > col_gap else " ") + b[4]
        out.append(s)
    return "\n".join(out)


def extract_tables(pdf: fitz.Document) -> list[ExtractedTable]:
    tables: list[ExtractedTable] = []
    for pno, page in enumerate(pdf, start=1):
        captions = _caption_blocks(page, TABLE_CAPTION_RE)
        if not _page_may_have_table(page, captions):
            continue
        try:
            found = [t for t in page.find_tables(strategy="lines").tables if t.row_count >= 2 and t.col_count >= 2]
        except Exception:
            found = []
        for t in found:
            rect = fitz.Rect(t.bbox) + (-2, -2, 2, 2)
            text = _table_rows_text(page, rect).strip()
            if text:
                tables.append(ExtractedTable(page=pno, markdown=text, caption=_nearest_caption(rect, captions, 80)))
    return tables


# -------------------------------------------------------------- images
def _save_clip(page: fitz.Page, rect: fitz.Rect, out_path: Path) -> None:
    pix = page.get_pixmap(matrix=fitz.Matrix(RENDER_ZOOM, RENDER_ZOOM), clip=rect, alpha=False)
    pix.save(out_path)


def _figure_regions(page: fitz.Page) -> list[tuple[fitz.Rect, str]]:
    """Zones de figures : images matricielles + dessins vectoriels proches d'une légende."""
    captions = _caption_blocks(page, FIGURE_CAPTION_RE)
    regions: list[fitz.Rect] = []

    # 1) Images matricielles intégrées (photos, captures, schémas scannés)
    for info in page.get_image_info():
        r = fitz.Rect(info["bbox"]) & page.rect
        if r.width >= MIN_IMAGE_SIDE and r.height >= MIN_IMAGE_SIDE:
            regions.append(r)

    # 2) Figures vectorielles : on regroupe les tracés en amas
    try:
        clusters = page.cluster_drawings()
    except Exception:
        clusters = []
    for r in clusters:
        r = fitz.Rect(r) & page.rect
        if r.width >= MIN_IMAGE_SIDE and r.height >= MIN_IMAGE_SIDE:
            regions.append(r)

    # Fusion des zones qui se chevauchent
    merged: list[fitz.Rect] = []
    for r in sorted(regions, key=lambda x: (x.y0, x.x0)):
        for i, m in enumerate(merged):
            if m.intersects(r + (-5, -5, 5, 5)):
                merged[i] = fitz.Rect(m) | r
                break
        else:
            merged.append(fitz.Rect(r))

    # On associe à chaque zone sa légende ; une petite marge garde les étiquettes du bord.
    result = []
    for m in merged:
        caption = _nearest_caption(m, captions)
        # Les traits seuls (ex. filets de tableaux) sans légende "Figure" sont ignorés
        is_raster = any(m.contains(fitz.Rect(i["bbox"])) for i in page.get_image_info())
        if not caption and not is_raster:
            continue
        result.append((m + (-4, -4, 4, 4), caption))
    return result


def extract_images(pdf: fitz.Document, doc_id: str) -> list[ExtractedImage]:
    out_dir = settings.images_dir / doc_id
    out_dir.mkdir(parents=True, exist_ok=True)
    images: list[ExtractedImage] = []
    for pno, page in enumerate(pdf, start=1):
        for idx, (rect, caption) in enumerate(_figure_regions(page), start=1):
            path = out_dir / f"p{pno:03d}_img{idx}.png"
            _save_clip(page, rect, path)
            images.append(ExtractedImage(page=pno, path=path, caption=caption, bbox=tuple(rect)))
    return images


# ---------------------------------------------------------------- tout
def parse_pdf(path: str | Path, doc_name: str | None = None) -> ParsedPDF:
    path = Path(path)
    doc_id = file_doc_id(path)
    doc_name = doc_name or path.name
    with fitz.open(path) as pdf:
        parsed = ParsedPDF(doc_id=doc_id, doc_name=doc_name, path=path, num_pages=pdf.page_count)
        parsed.tables = extract_tables(pdf)
        parsed.images = extract_images(pdf, doc_id)
    parsed.pages = extract_text(path, doc_id, doc_name)
    return parsed
