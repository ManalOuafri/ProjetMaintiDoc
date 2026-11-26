"""Configuration centrale de MaintiDoc (lue depuis le fichier .env)."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")

# Logs HTTP (Ollama, HuggingFace) trop verbeux en INFO
for _name in ("httpx", "httpcore", "huggingface_hub", "sentence_transformers"):
    logging.getLogger(_name).setLevel(logging.WARNING)
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")


@dataclass(frozen=True)
class Settings:
    # PostgreSQL / PGVector
    pg_host: str = os.getenv("PG_HOST", "localhost")
    pg_port: int = int(os.getenv("PG_PORT", "5434"))
    pg_user: str = os.getenv("PG_USER", "maintidoc")
    pg_password: str = os.getenv("PG_PASSWORD", "maintidoc")
    pg_database: str = os.getenv("PG_DATABASE", "maintidoc")

    # Tables PGVector (LlamaIndex préfixe automatiquement par "data_")
    text_table: str = "maintidoc_text"     # chunks de texte, tableaux, descriptions d'images
    image_table: str = "maintidoc_images"  # vecteurs CLIP des images

    # Ollama
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    llm_model: str = os.getenv("LLM_MODEL", "qwen2.5:7b")
    vlm_model: str = os.getenv("VLM_MODEL", "qwen2.5vl:3b")

    # Embeddings
    embed_backend: str = os.getenv("EMBED_BACKEND", "ollama")  # "ollama" (GPU) ou "huggingface" (CPU)
    ollama_embed_model: str = os.getenv("OLLAMA_EMBED_MODEL", "bge-m3")
    text_embed_model: str = os.getenv("TEXT_EMBED_MODEL", "BAAI/bge-m3")
    text_embed_dim: int = 1024  # dimension de BGE-M3
    clip_model: str = os.getenv("CLIP_MODEL", "openai/clip-vit-base-patch32")
    clip_dim: int = 512  # dimension de CLIP ViT-B/32
    embed_device: str = os.getenv("EMBED_DEVICE", "cpu")

    # Découpage du texte
    chunk_size: int = 512
    chunk_overlap: int = 64

    # Dossiers de travail
    data_dir: Path = ROOT_DIR / "data"

    @property
    def uploads_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def images_dir(self) -> Path:
        return self.data_dir / "images"


settings = Settings()
settings.uploads_dir.mkdir(parents=True, exist_ok=True)
settings.images_dir.mkdir(parents=True, exist_ok=True)
