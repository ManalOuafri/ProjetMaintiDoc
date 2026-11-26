"""Jeu de questions de démonstration : agent + RAG classique.

Usage : uv run python scripts/eval_questions.py > data/eval.md
Les sorties servent d'exemples dans docs/NOTE_TECHNIQUE.md.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from maintidoc.agent import MaintiDocAgent  # noqa: E402
from maintidoc.rag import answer  # noqa: E402

QUESTIONS = [
    ("figure", "Que représente la figure 1 du document ?"),
    ("tableau", "Quel score BLEU obtient le Transformer (big) sur la traduction anglais-allemand, et combien a coûté son entraînement ?"),
    ("texte", "Pourquoi divise-t-on le produit scalaire par la racine carrée de dk dans l'attention ?"),
    ("texte+tableau", "Combien de têtes d'attention utilise le modèle de base, et quelles sont les dimensions dk et dv ?"),
    ("hors corpus", "Quelle est la pression nominale de la pompe hydraulique P-200 ?"),
]


def main() -> None:
    agent = MaintiDocAgent()
    for kind, q in QUESTIONS:
        print(f"\n## [{kind}] {q}\n", flush=True)
        t0 = time.time()
        res = agent.ask(q, thread_id=f"eval-{kind}")
        print(f"**Agent** ({time.time() - t0:.0f} s) — outils : " + ", ".join(f"`{s['tool']}({s['args']})`" for s in res["steps"]))
        print("\n" + res["answer"] + "\n")
        print("Sources : " + ", ".join(sorted({h.citation for h in res["hits"]})), flush=True)

    q = QUESTIONS[1][1]
    print(f"\n## [RAG classique] {q}\n")
    t0 = time.time()
    res = answer(q)
    print(f"**RAG classique** ({time.time() - t0:.0f} s)\n\n{res.answer}\n")
    print("Sources : " + ", ".join(sorted({h.citation for h in res.hits})), flush=True)


if __name__ == "__main__":
    main()
