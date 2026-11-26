"""Interface Streamlit de MaintiDoc (étape 5).

Lancement :  uv run streamlit run app.py
- Barre latérale : chargement / indexation d'un PDF, documents indexés, réglages.
- Zone principale : conversation (RAG agentique ou RAG classique) avec sources,
  tableaux et figures affichés sous chaque réponse.
"""

from __future__ import annotations

import uuid
from pathlib import Path

import streamlit as st

from maintidoc.config import settings

st.set_page_config(page_title="MaintiDoc", page_icon="🛠️", layout="wide")


# ------------------------------------------------------------ helpers
@st.cache_resource(show_spinner="Chargement des modèles d'embedding (BGE-M3, CLIP)…")
def warm_up_models():
    from maintidoc.embeddings import get_clip_encoder, get_text_embed_model

    get_text_embed_model()
    get_clip_encoder()
    return True


def get_agent(model: str, doc_id: str | None):
    from maintidoc.agent import MaintiDocAgent

    key = (model, doc_id)
    if st.session_state.get("agent_key") != key:
        st.session_state.agent = MaintiDocAgent(model=model, doc_id=doc_id)
        st.session_state.agent_key = key
        st.session_state.thread_id = str(uuid.uuid4())
    return st.session_state.agent


def render_sources(hits, images=None):
    """Affiche les sources : passages, tableaux, figures (sans doublons)."""
    images = images or []
    seen = set()
    uniq = []
    for h in hits + images:
        key = (h.doc_name, h.page, h.type, h.image_path or h.text[:80])
        if key not in seen:
            seen.add(key)
            uniq.append(h)
    if not uniq:
        return
    figs = [h for h in uniq if h.type == "image" and h.image_path and Path(h.image_path).exists()]
    tables = [h for h in uniq if h.type == "table"]
    texts = [h for h in uniq if h.type == "text"]

    with st.expander(f"📚 Sources ({len(uniq)})", expanded=False):
        for h in texts:
            st.markdown(f"**{h.citation}** · score {h.score:.2f}")
            st.caption(h.text[:600] + ("…" if len(h.text) > 600 else ""))
        for h in tables:
            st.markdown(f"**{h.citation}** · score {h.score:.2f}")
            st.markdown(h.text)
    if figs:
        cols = st.columns(min(len(figs), 3))
        for col, h in zip(cols * 3, figs[:6]):
            with col:
                st.image(h.image_path, caption=f"{h.citation} {h.caption[:90]}", use_container_width=True)


# ------------------------------------------------------------ sidebar
st.sidebar.title("🛠️ MaintiDoc")
st.sidebar.caption("Partie I — RAG multimodal agentique")

uploaded = st.sidebar.file_uploader("Charger un PDF technique", type=["pdf"])
if uploaded and st.sidebar.button("Indexer le document", type="primary", use_container_width=True):
    from maintidoc.indexing import index_pdf

    warm_up_models()
    dest = settings.uploads_dir / uploaded.name
    dest.write_bytes(uploaded.getbuffer())
    bar = st.sidebar.progress(0.0, text="Démarrage…")
    try:
        report = index_pdf(dest, doc_name=uploaded.name, progress=lambda msg, frac: bar.progress(min(max(frac, 0.0), 1.0), text=msg))
        st.sidebar.success(
            f"{report.doc_name} indexé : {report.num_pages} pages, {report.text_chunks} chunks, "
            f"{report.tables} tableaux, {report.images} figures."
        )
    except Exception as e:  # message clair pour l'utilisateur (DB arrêtée, Ollama absent…)
        st.sidebar.error(f"Échec de l'indexation : {e}")

from maintidoc.vector_store import delete_document, list_documents  # noqa: E402

try:
    docs = list_documents()
except Exception as e:
    docs = []
    st.sidebar.error(f"PGVector inaccessible ({e}). Lancez `docker compose up -d`.")

st.sidebar.subheader("Documents indexés")
doc_options = {"Tous les documents": None}
for d in docs:
    st.sidebar.markdown(f"- **{d['doc_name']}** — {d['text']} chunks · {d['table']} tableaux · {d['image']} figures")
    doc_options[d["doc_name"]] = d["doc_id"]
if not docs:
    st.sidebar.info("Aucun document : chargez un PDF puis cliquez sur « Indexer ».")

scope = st.sidebar.selectbox("Rechercher dans", list(doc_options))
doc_id = doc_options[scope]

st.sidebar.subheader("Réglages")
mode = st.sidebar.radio("Mode", ["Agent (LangChain)", "RAG classique"], help="L'agent choisit lui-même ses outils de recherche ; le RAG classique suit toujours le même pipeline.")
model = st.sidebar.text_input("Modèle Ollama", value=settings.llm_model)

c1, c2 = st.sidebar.columns(2)
if c1.button("Nouvelle conversation", use_container_width=True):
    st.session_state.history = []
    st.session_state.pop("agent_key", None)
    st.rerun()
if doc_id and c2.button("Supprimer le doc", use_container_width=True):
    delete_document(doc_id)
    st.rerun()

# ------------------------------------------------------------ chat
st.title("Assistant documentaire de maintenance")
st.caption(
    "Les réponses sont fondées sur les documents indexés et citent leurs sources. "
    "MaintiDoc ne pilote aucun équipement et ne remplace pas un technicien habilité."
)

if "history" not in st.session_state:
    st.session_state.history = []

for turn in st.session_state.history:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])
        if turn["role"] == "assistant":
            if turn.get("steps"):
                with st.expander("🧠 Raisonnement de l'agent (outils appelés)"):
                    for s in turn["steps"]:
                        st.code(f"{s['tool']}({s['args']})")
            render_sources(turn.get("hits", []), turn.get("images", []))

question = st.chat_input("Posez votre question (ex. : Que représente la figure 1 ?)")
if question:
    st.session_state.history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    warm_up_models()
    with st.chat_message("assistant"):
        try:
            if mode.startswith("Agent"):
                with st.spinner("L'agent recherche dans les documents…"):
                    agent = get_agent(model, doc_id)
                    res = agent.ask(question, thread_id=st.session_state.thread_id)
                st.markdown(res["answer"])
                turn = {"role": "assistant", "content": res["answer"], "steps": res["steps"], "hits": res["hits"], "images": []}
                if res["steps"]:
                    with st.expander("🧠 Raisonnement de l'agent (outils appelés)"):
                        for s in res["steps"]:
                            st.code(f"{s['tool']}({s['args']})")
            else:
                from maintidoc.rag import answer_stream

                with st.spinner("Recherche des passages pertinents…"):
                    stream, hits, images = answer_stream(question, doc_id=doc_id, model=model)
                text = st.write_stream(stream)
                turn = {"role": "assistant", "content": text, "hits": hits, "images": images}
            render_sources(turn.get("hits", []), turn.get("images", []))
        except Exception as e:
            turn = {"role": "assistant", "content": f"⚠️ Erreur : {e}"}
            st.error(turn["content"])
    st.session_state.history.append(turn)
