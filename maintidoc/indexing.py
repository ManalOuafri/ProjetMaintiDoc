"""Pipeline d'indexation : PDF -> chunks / tableaux / images -> embeddings -> PGVector.

Étapes :
1. `parse_pdf` extrait texte (LlamaIndex), tableaux et images (PyMuPDF).
2. Le texte est découpé en chunks (`SentenceSplitter`, 512 tokens, chevauchement 64).
3. Chaque image reçoit une description générée par Qwen2.5-VL.
4. Chunks + tableaux + descriptions sont vectorisés avec BGE-M3 -> table texte.
5. Les images sont vectorisées avec CLIP -> table images.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import NodeRelationship, RelatedNodeInfo, TextNode

from .captioning import describe_image
from .config import settings
from .embeddings import get_clip_encoder, get_text_embed_model
from .pdf_parser import ParsedPDF, parse_pdf
from .vector_store import delete_document, get_image_store, get_text_store

ProgressFn = Callable[[str, float], None]


@dataclass
class IndexReport:
    doc_id: str
    doc_name: str
    num_pages: int
    text_chunks: int
    tables: int
    images: int


def _link_to_doc(node: TextNode, doc_id: str) -> TextNode:
    # ref_doc_id = doc_id : permet de supprimer/remplacer tout un document d'un coup
    node.relationships[NodeRelationship.SOURCE] = RelatedNodeInfo(node_id=doc_id)
    return node


def build_text_nodes(parsed: ParsedPDF) -> list[TextNode]:
    splitter = SentenceSplitter(chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap)
    nodes = splitter.get_nodes_from_documents(parsed.pages)
    return [_link_to_doc(n, parsed.doc_id) for n in nodes]


def build_table_nodes(parsed: ParsedPDF) -> list[TextNode]:
    nodes = []
    for t in parsed.tables:
        text = f"{t.caption}\n\n{t.markdown}" if t.caption else t.markdown
        nodes.append(
            _link_to_doc(
                TextNode(
                    text=text,
                    metadata={"doc_id": parsed.doc_id, "doc_name": parsed.doc_name, "page": t.page, "type": "table", "caption": t.caption},
                ),
                parsed.doc_id,
            )
        )
    return nodes


def build_image_nodes(parsed: ParsedPDF, progress: ProgressFn | None = None) -> tuple[list[TextNode], list[TextNode]]:
    """Retourne (nœuds description pour la table texte, nœuds CLIP pour la table images)."""
    desc_nodes, clip_nodes = [], []
    for i, img in enumerate(parsed.images, start=1):
        if progress:
            progress(f"Description de l'image {i}/{len(parsed.images)} (page {img.page}) avec {settings.vlm_model}…", i / max(len(parsed.images), 1))
        description = describe_image(img.path, img.caption)
        meta = {
            "doc_id": parsed.doc_id,
            "doc_name": parsed.doc_name,
            "page": img.page,
            "type": "image",
            "caption": img.caption,
            "image_path": str(img.path),
        }
        text = f"[Figure, page {img.page}] {img.caption}\nDescription: {description}"
        desc_nodes.append(_link_to_doc(TextNode(text=text, metadata=dict(meta)), parsed.doc_id))
        clip_nodes.append(_link_to_doc(TextNode(text=text, metadata=dict(meta)), parsed.doc_id))
    return desc_nodes, clip_nodes


def index_pdf(path: str | Path, doc_name: str | None = None, progress: ProgressFn | None = None) -> IndexReport:
    def step(msg: str, frac: float = 0.0):
        if progress:
            progress(msg, frac)

    step("Analyse du PDF (texte, tableaux, images)…")
    parsed = parse_pdf(path, doc_name)

    step("Suppression d'une éventuelle version précédente du document…")
    delete_document(parsed.doc_id)

    text_nodes = build_text_nodes(parsed)
    table_nodes = build_table_nodes(parsed)
    desc_nodes, clip_nodes = build_image_nodes(parsed, progress)

    # --- Embeddings texte (BGE-M3) -> PGVector
    all_text_nodes = text_nodes + table_nodes + desc_nodes
    step(f"Embeddings BGE-M3 de {len(all_text_nodes)} éléments…")
    embed_model = get_text_embed_model()
    vectors = embed_model.get_text_embedding_batch([n.get_content() for n in all_text_nodes], show_progress=False)
    for n, v in zip(all_text_nodes, vectors):
        n.embedding = v
    get_text_store().add(all_text_nodes)

    # --- Embeddings images (CLIP) -> PGVector
    if clip_nodes:
        step(f"Embeddings CLIP de {len(clip_nodes)} images…")
        clip_vectors = get_clip_encoder().embed_images([n.metadata["image_path"] for n in clip_nodes])
        for n, v in zip(clip_nodes, clip_vectors):
            n.embedding = v
        get_image_store().add(clip_nodes)

    step("Indexation terminée.", 1.0)
    return IndexReport(
        doc_id=parsed.doc_id,
        doc_name=parsed.doc_name,
        num_pages=parsed.num_pages,
        text_chunks=len(text_nodes),
        tables=len(table_nodes),
        images=len(clip_nodes),
    )
