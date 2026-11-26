# MaintiDoc — Partie I : RAG multimodal agentique

Assistant documentaire pour techniciens de maintenance. Cette première partie
indexe un PDF (texte, tableaux, images) dans **PGVector** et répond aux questions
avec des **références vérifiables** (document, page, type d'élément), grâce à un
LLM open source local orchestré par un **agent LangChain**.

> MaintiDoc ne pilote aucun équipement et ne remplace pas un technicien habilité.

## Architecture

```
                 ┌──────────── INDEXATION ────────────────────────────────────────────┐
PDF ─► LlamaIndex (PyMuPDFReader) ─► texte ─► SentenceSplitter (512 tk) ─┐
   └─► PyMuPDF find_tables ─► tableaux (Markdown) ──────────────────────┼─► BGE-M3 ─► PGVector  data_maintidoc_text
   └─► PyMuPDF images + figures vectorielles ─► PNG ─► Qwen2.5-VL ─► description ┘
                                               └──► CLIP ViT-B/32 ─────────────────► PGVector  data_maintidoc_images
                 └────────────────────────────────────────────────────────────────────┘
                 ┌──────────── QUESTION / RÉPONSE ─────────────────────────────────────┐
Streamlit ─► Agent LangChain (ReAct + mémoire) ─► outils : search_documents
                                                         search_tables
                                                         search_figures (descriptions + CLIP)
                                                         list_documents
          ◄── réponse sourcée + passages, tableaux et figures ◄── Qwen2.5 7B (Ollama)
```

| Brique | Choix | Rôle |
|---|---|---|
| Parsing | LlamaIndex `PyMuPDFReader` + PyMuPDF | texte par page, tableaux, images/figures |
| Découpage | LlamaIndex `SentenceSplitter` (512 / 64) | chunks avec métadonnées (doc, page, type) |
| Embeddings texte | `bge-m3` (1024 d, multilingue) via Ollama | question FR ↔ document EN |
| Images | `openai/clip-vit-base-patch32` (512 d) | recherche texte → image |
| Description d'images | `qwen2.5vl:3b` via Ollama | rend les figures interrogeables en texte |
| Base vectorielle | PostgreSQL 16 + pgvector (index HNSW, cosinus) | vecteurs + métadonnées JSONB |
| LLM | `qwen2.5:7b` via Ollama | génération et raisonnement de l'agent |
| Agent | LangChain `create_agent` + `InMemorySaver` | choix des outils, mémoire de conversation |
| Interface | Streamlit | chargement PDF, questions, réponses, sources |

## Structure

```
maintidoc/
  config.py        configuration (.env)
  pdf_parser.py    extraction texte / tableaux / images
  captioning.py    descriptions d'images (Qwen2.5-VL)
  embeddings.py    BGE-M3 + encodeur CLIP
  vector_store.py  tables PGVector (LlamaIndex PGVectorStore)
  indexing.py      pipeline d'indexation complet
  retrieval.py     recherche vectorielle + formatage du contexte
  llm.py           Qwen via Ollama (LangChain ChatOllama)
  rag.py           RAG classique (étape 4)
  agent.py         RAG agentique (étape 6)
app.py             interface Streamlit (étape 5)
scripts/ingest.py  indexation en ligne de commande
scripts/ask.py     questions en ligne de commande
scripts/eval_questions.py  jeu de questions de démonstration
docs/NOTE_TECHNIQUE.md
```

## Installation

Prérequis : Docker Desktop, [Ollama](https://ollama.com), [uv](https://docs.astral.sh/uv/), Python 3.11–3.13.

```bash
# 1. Base vectorielle PostgreSQL + pgvector (port 5434)
cp .env.example .env
docker compose up -d

# 2. Modèles Ollama
ollama pull qwen2.5:7b
ollama pull qwen2.5vl:3b
ollama pull bge-m3

# 3. Dépendances Python
uv sync            # (derrière un proxy/antivirus : uv sync --native-tls)
```

CLIP est téléchargé automatiquement depuis HuggingFace au premier lancement.
Par défaut, BGE-M3 est servi par Ollama (GPU). Avec `EMBED_BACKEND=huggingface`
dans `.env`, il tourne via sentence-transformers sur CPU (environ 8 s par chunk
sur un portable). Changer de backend impose de ré-indexer les documents.

## Utilisation

```bash
# Interface web
uv run streamlit run app.py

# Ou en ligne de commande
uv run python scripts/ingest.py "docs/attention is all you need.pdf"
uv run python scripts/ask.py "Que représente la figure 1 ?"
uv run python scripts/ask.py --rag "Quel score BLEU obtient le Transformer big en anglais-allemand ?"
```

Vérifier le contenu de la base :

```bash
docker exec -it maintidoc-pgvector psql -U maintidoc -d maintidoc \
  -c "SELECT metadata_->>'type' AS type, COUNT(*) FROM data_maintidoc_text GROUP BY 1;"
```
