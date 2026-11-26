"""Accès aux deux tables PGVector via LlamaIndex.

- `data_maintidoc_text`   : vecteurs BGE-M3 (1024) des chunks de texte, des
                            tableaux (Markdown) et des descriptions d'images.
- `data_maintidoc_images` : vecteurs CLIP (512) des images.

Les métadonnées (doc_id, doc_name, page, type, légende, chemin de l'image)
sont stockées en JSONB dans la colonne `metadata_`.
"""

from __future__ import annotations

from functools import lru_cache

import psycopg2
from llama_index.vector_stores.postgres import PGVectorStore

from .config import settings


def _make_store(table: str, dim: int) -> PGVectorStore:
    return PGVectorStore.from_params(
        host=settings.pg_host,
        port=str(settings.pg_port),
        database=settings.pg_database,
        user=settings.pg_user,
        password=settings.pg_password,
        table_name=table,
        embed_dim=dim,
        use_jsonb=True,
        hnsw_kwargs={"hnsw_m": 16, "hnsw_ef_construction": 64, "hnsw_ef_search": 40, "hnsw_dist_method": "vector_cosine_ops"},
    )


@lru_cache(maxsize=1)
def get_text_store() -> PGVectorStore:
    return _make_store(settings.text_table, settings.text_embed_dim)


@lru_cache(maxsize=1)
def get_image_store() -> PGVectorStore:
    return _make_store(settings.image_table, settings.clip_dim)


def _connect():
    return psycopg2.connect(
        host=settings.pg_host,
        port=settings.pg_port,
        dbname=settings.pg_database,
        user=settings.pg_user,
        password=settings.pg_password,
    )


def list_documents() -> list[dict]:
    """Documents indexés, avec le nombre d'éléments par type."""
    sql = f"""
        SELECT metadata_->>'doc_id', metadata_->>'doc_name', metadata_->>'type', COUNT(*)
        FROM data_{settings.text_table}
        GROUP BY 1, 2, 3 ORDER BY 2
    """
    docs: dict[str, dict] = {}
    try:
        with _connect() as conn, conn.cursor() as cur:
            cur.execute(sql)
            for doc_id, name, typ, n in cur.fetchall():
                d = docs.setdefault(doc_id, {"doc_id": doc_id, "doc_name": name, "text": 0, "table": 0, "image": 0})
                d[typ] = n
    except psycopg2.errors.UndefinedTable:
        return []
    return list(docs.values())


def delete_document(doc_id: str) -> None:
    """Supprime toutes les entrées d'un document dans les deux tables."""
    get_text_store().delete(ref_doc_id=doc_id)
    get_image_store().delete(ref_doc_id=doc_id)
