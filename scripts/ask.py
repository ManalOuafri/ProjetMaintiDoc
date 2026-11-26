"""Pose une question en ligne de commande (agent ou RAG classique).

Usage :
  uv run python scripts/ask.py "Que représente la figure 1 ?"
  uv run python scripts/ask.py --rag "Quel score BLEU obtient le Transformer big ?"
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> None:
    args = sys.argv[1:]
    use_rag = "--rag" in args
    args = [a for a in args if a != "--rag"]
    if not args:
        print(__doc__)
        sys.exit(1)
    question = " ".join(args)
    t0 = time.time()

    if use_rag:
        from maintidoc.rag import answer

        res = answer(question)
        print(res.answer)
        hits = res.hits + res.images
    else:
        from maintidoc.agent import MaintiDocAgent

        res = MaintiDocAgent().ask(question)
        for s in res["steps"]:
            print(f"  → {s['tool']}({s['args']})")
        print(res["answer"])
        hits = res["hits"]

    print("\nSources :")
    for h in hits:
        print(f"  {h.citation} score={h.score:.2f} {h.image_path}")
    print(f"\n({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
