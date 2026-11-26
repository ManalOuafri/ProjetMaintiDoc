"""Génération de descriptions textuelles des images avec un VLM local (Qwen2.5-VL via Ollama).

CLIP sait *représenter* une image (vecteur) mais pas la *décrire* : on utilise donc
un modèle vision-langage pour produire une description, qui est ensuite indexée
comme du texte (BGE-M3) et devient interrogeable par le RAG.
"""

from __future__ import annotations

from pathlib import Path

import ollama

from .config import settings

CAPTION_PROMPT = (
    "You are documenting a figure extracted from a technical PDF.\n"
    "Caption found in the document (may be empty): {caption}\n\n"
    "Describe the figure precisely in 4 to 8 sentences: what type of figure it is "
    "(diagram, schema, chart, photo, table...), every labelled component or block, "
    "how the elements are connected (arrows, flows, order), and any visible text, "
    "numbers or units. Do not invent information that is not visible."
)


def describe_image(image_path: str | Path, caption: str = "") -> str:
    """Retourne une description textuelle de l'image (en anglais, comme le document)."""
    client = ollama.Client(host=settings.ollama_base_url)
    response = client.chat(
        model=settings.vlm_model,
        messages=[
            {
                "role": "user",
                "content": CAPTION_PROMPT.format(caption=caption or "(none)"),
                "images": [str(image_path)],
            }
        ],
        options={"temperature": 0.1, "num_predict": 400},
    )
    return response["message"]["content"].strip()
