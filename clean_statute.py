import json
import sys
from pathlib import Path

# ---------- configuration ----------
KEEP_TOP = {
    "_id",
    "title",
    "titleNum",
    "chapterNum",
    "subChapterNum",
    "sectionNum",
    "text",
    "sections"          # <- keep this so we can recurse
}

KEEP_SEC = {
    "_id",
    "title",
    "titleNum",
    "chapterNum",
    "subChapterNum",
    "sectionNum",
    "text"
}
# -----------------------------------

def clean_article(article: dict) -> dict:
    """Return a filtered copy of an article, including cleaned sections."""
    cleaned = {k: article[k] for k in KEEP_TOP if k in article}

    # Recursively clean sections, if present
    if "sections" in cleaned and isinstance(cleaned["sections"], list):
        cleaned["sections"] = [
            {k: sec[k] for k in KEEP_SEC if k in sec}
            for sec in article.get("sections", [])
        ]
    return cleaned

def main(src_path: Path, dst_path: Path):
    with src_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # Input may be a list or a single dict
    if isinstance(data, list):
        cleaned = [clean_article(art) for art in data]
    else:
        cleaned = clean_article(data)

    with dst_path.open("w", encoding="utf-8") as f:
        json.dump(cleaned, f, ensure_ascii=False, indent=2)

    print(f"Cleaned JSON written to {dst_path}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python clean_json.py <raw.json> <clean.json>")
        sys.exit(1)
    main(Path(sys.argv[1]), Path(sys.argv[2]))
