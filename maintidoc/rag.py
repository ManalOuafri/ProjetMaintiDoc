"""Pipeline RAG « classique » (étape 4) : récupération -> prompt -> génération.

Contrairement à l'agent, ce pipeline suit toujours le même chemin :
il récupère des passages texte, des tableaux et des descriptions de figures,
les injecte dans le prompt, puis demande au LLM de répondre uniquement
à partir de ce contexte.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

from langchain_core.messages import HumanMessage, SystemMessage

from .llm import get_llm
from .retrieval import Hit, format_context, search_images, search_text

SYSTEM_PROMPT = """Tu es MaintiDoc, un assistant documentaire pour techniciens de maintenance.
Réponds UNIQUEMENT à partir du contexte fourni entre les balises <context>.
- Réponds dans la langue de la question (français par défaut), même si le document est en anglais.
- Après chaque affirmation, cite la source entre crochets, exactement comme indiqué dans l'attribut ref (ex. [doc.pdf, p. 3, texte]).
- Si la réponse n'est pas dans le contexte, réponds : « Je ne sais pas : cette information n'est pas disponible dans les documents indexés. »
- Ne pilote aucun équipement et ne présente jamais un diagnostic comme certain : tu assistes un technicien habilité, tu ne le remplaces pas."""

USER_TEMPLATE = """<context>
{context}
</context>

<question>
{question}
</question>"""


@dataclass
class RAGResult:
    answer: str
    hits: list[Hit]
    images: list[Hit]


def retrieve(question: str, doc_id: str | None = None, k_text: int = 5, k_tables: int = 2, k_figures: int = 2) -> tuple[list[Hit], list[Hit]]:
    hits = search_text(question, k=k_text, types=["text"], doc_id=doc_id)
    hits += search_text(question, k=k_tables, types=["table"], doc_id=doc_id)
    hits += search_text(question, k=k_figures, types=["image"], doc_id=doc_id)
    images = search_images(question, k=2, doc_id=doc_id)
    return hits, images


def _messages(question: str, hits: list[Hit]):
    return [
        SystemMessage(SYSTEM_PROMPT),
        HumanMessage(USER_TEMPLATE.format(context=format_context(hits), question=question)),
    ]


def answer(question: str, doc_id: str | None = None, model: str | None = None) -> RAGResult:
    hits, images = retrieve(question, doc_id)
    resp = get_llm(model).invoke(_messages(question, hits))
    return RAGResult(answer=resp.content, hits=hits, images=images)


def answer_stream(question: str, doc_id: str | None = None, model: str | None = None) -> tuple[Iterator[str], list[Hit], list[Hit]]:
    """Version streaming pour l'interface : renvoie (générateur de texte, passages, images)."""
    hits, images = retrieve(question, doc_id)
    stream = (chunk.content for chunk in get_llm(model).stream(_messages(question, hits)))
    return stream, hits, images
