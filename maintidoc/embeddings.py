"""Modèles d'embedding open source.

- Texte : BGE-M3 (multilingue, 1024 dimensions) via LlamaIndex.
  Multilingue => une question en français retrouve un passage en anglais.
  Deux backends (variable EMBED_BACKEND) :
    * "ollama"      : BGE-M3 servi par Ollama, sur GPU (rapide, par défaut) ;
    * "huggingface" : BGE-M3 via sentence-transformers (CPU, ~8 s / chunk sur un portable).
  Attention : changer de backend impose de ré-indexer les documents.
- Images : CLIP ViT-B/32 (512 dimensions) via HuggingFace transformers.
  CLIP place images et textes dans le même espace vectoriel, ce qui permet
  la recherche texte -> image.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import torch
from llama_index.core.base.embeddings.base import BaseEmbedding
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from .config import settings


@lru_cache(maxsize=1)
def get_text_embed_model() -> BaseEmbedding:
    """Modèle d'embedding texte (chargé une seule fois)."""
    if settings.embed_backend == "ollama":
        from llama_index.embeddings.ollama import OllamaEmbedding

        return OllamaEmbedding(
            model_name=settings.ollama_embed_model,
            base_url=settings.ollama_base_url,
            embed_batch_size=8,
        )
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding

    return HuggingFaceEmbedding(
        model_name=settings.text_embed_model,
        device=settings.embed_device,
        embed_batch_size=8,
    )


class ClipEncoder:
    """Encodeur CLIP : images et textes -> vecteurs normalisés de 512 dimensions."""

    def __init__(self, model_name: str = settings.clip_model, device: str = settings.embed_device):
        self.device = device
        self.model = CLIPModel.from_pretrained(model_name).to(device).eval()
        self.processor = CLIPProcessor.from_pretrained(model_name)

    @staticmethod
    def _normalize(t) -> list[list[float]]:
        # Selon la version de transformers, get_*_features renvoie un tenseur
        # ou un objet de sortie dont `pooler_output` contient la projection CLIP.
        if not isinstance(t, torch.Tensor):
            t = t.pooler_output
        t = t / t.norm(dim=-1, keepdim=True)
        return t.cpu().tolist()

    @torch.no_grad()
    def embed_images(self, paths: list[str | Path]) -> list[list[float]]:
        images = [Image.open(p).convert("RGB") for p in paths]
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        return self._normalize(self.model.get_image_features(**inputs))

    @torch.no_grad()
    def embed_text(self, text: str) -> list[float]:
        # CLIP est limité à 77 tokens et entraîné surtout en anglais.
        inputs = self.processor(text=[text], return_tensors="pt", padding=True, truncation=True).to(self.device)
        return self._normalize(self.model.get_text_features(**inputs))[0]


@lru_cache(maxsize=1)
def get_clip_encoder() -> ClipEncoder:
    return ClipEncoder()
