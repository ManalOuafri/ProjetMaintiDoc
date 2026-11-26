"""Recherche vectorielle dans PGVector.

- `search_text` : similarité cosinus BGE-M3 sur la table texte, avec filtre
  optionnel par type ("text", "table", "image" = descriptions d'images).
- `search_images` : recherche texte -> image avec CLIP sur la table images.
"""

from __future__ import annotations

from dataclasses import dataclass

from llama_index.core.vector_stores import FilterOperator, MetadataFilter, MetadataFilters, VectorStoreQuery

from .embeddings import get_clip_encoder, get_text_embed_model
from .vector_store import get_image_store, get_text_store


@dataclass
class Hit:
    text: str
    score: float
    doc_name: str
    page: int
    type: str
    caption: str = ""
    image_path: str = ""

    @property
    def citation(self) -> str:
        label = {"text": "texte", "table": "tableau", "image": "figure"}.get(self.type, self.type)
        return f"[{self.doc_name}, p. {self.page}, {label}]"


def _filters(types: list[str] | None, doc_id: str | None) -> MetadataFilters | None:
    f = []
    if types:
        f.append(MetadataFilter(key="type", value=types, operator=FilterOperator.IN))
    if doc_id:
        f.append(MetadataFilter(key="doc_id", value=doc_id, operator=FilterOperator.EQ))
    return MetadataFilters(filters=f) if f else None


def _to_hits(result) -> list[Hit]:
    hits = []
    for node, score in zip(result.nodes or [], result.similarities or []):
        m = node.metadata
        hits.append(
            Hit(
                text=node.get_content(),
                score=float(score),
                doc_name=m.get("doc_name", "?"),
                page=int(m.get("page", 0)),
                type=m.get("type", "text"),
                caption=m.get("caption", ""),
                image_path=m.get("image_path", ""),
            )
        )
    return hits


def search_text(query: str, k: int = 5, types: list[str] | None = None, doc_id: str | None = None) -> list[Hit]:
    embedding = get_text_embed_model().get_query_embedding(query)
    q = VectorStoreQuery(query_embedding=embedding, similarity_top_k=k, filters=_filters(types, doc_id))
    return _to_hits(get_text_store().query(q))


def search_images(query: str, k: int = 3, doc_id: str | None = None) -> list[Hit]:
    embedding = get_clip_encoder().embed_text(query)
    q = VectorStoreQuery(query_embedding=embedding, similarity_top_k=k, filters=_filters(None, doc_id))
    return _to_hits(get_image_store().query(q))


def format_context(hits: list[Hit], max_chars: int = 1500) -> str:
    """Contexte lisible par le LLM, chaque passage précédé de sa source."""
    blocks = []
    for i, h in enumerate(hits, start=1):
        text = h.text if len(h.text) <= max_chars else h.text[:max_chars] + " […]"
        blocks.append(f"<source id={i} ref=\"{h.citation}\">\n{text}\n</source>")
    return "\n\n".join(blocks) if blocks else "(aucun passage trouvé)"
