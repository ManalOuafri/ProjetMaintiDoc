"""Indexe un PDF en ligne de commande.

Usage : uv run python scripts/ingest.py "docs/attention is all you need.pdf"
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from maintidoc.indexing import index_pdf  # noqa: E402


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    t0 = time.time()
    report = index_pdf(sys.argv[1], progress=lambda msg, frac: print(f"  · {msg}", flush=True))
    print(
        f"\n✔ {report.doc_name} (id {report.doc_id}) : {report.num_pages} pages, "
        f"{report.text_chunks} chunks texte, {report.tables} tableaux, {report.images} figures "
        f"— {time.time() - t0:.0f}s"
    )


if __name__ == "__main__":
    main()
