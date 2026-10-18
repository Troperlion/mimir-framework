#!/usr/bin/env python3
"""Vérifie les liens relatifs de la documentation Markdown active."""

from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs"
LINK = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")


def clean_target(raw: str) -> str:
    target = raw.strip()
    if target.startswith("<") and target.endswith(">"):
        target = target[1:-1]
    # Un titre Markdown facultatif suit parfois l'URL entre guillemets.
    return target.split(' "', 1)[0].split(" '", 1)[0]


def main() -> int:
    failures: list[str] = []
    checked = 0
    pages = [ROOT / "README.md"]
    pages.extend(
        path for path in DOCS.rglob("*.md")
        if "docs_archive" not in path.parts
    )

    for page in sorted(pages):
        text = page.read_text(encoding="utf-8")
        for match in LINK.finditer(text):
            target = clean_target(match.group(1))
            if (not target or target.startswith(
                    ("#", "http://", "https://", "mailto:", "const "))):
                continue
            path_text = target.split("#", 1)[0]
            if not path_text:
                continue
            checked += 1
            resolved = (page.parent / path_text).resolve()
            if not resolved.exists():
                line = text.count("\n", 0, match.start()) + 1
                failures.append(f"{page.relative_to(ROOT)}:{line}: {target}")

    if failures:
        print("Liens locaux manquants:", file=sys.stderr)
        for failure in failures:
            print(f"  {failure}", file=sys.stderr)
        return 1

    print(f"Documentation vérifiée: {len(pages)} pages, {checked} liens locaux valides")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
