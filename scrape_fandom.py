#!/usr/bin/env python3
"""
Riordan Fandom Wiki — Creature Image Scraper
=============================================

Downloads images from every page in a given category on the Riordan fandom
wiki, using the MediaWiki API (NOT HTML scraping — cleaner and more polite).
Organizes the output as one folder per creature.

Usage
-----
    pip install requests
    python scrape_riordan_creatures.py
    python scrape_riordan_creatures.py --lang de --category "Kategorie:Monster"
    python scrape_riordan_creatures.py --lang en --category "Category:Monsters"
    python scrape_riordan_creatures.py --limit 5   # dry-runnish test

Finding the right category
--------------------------
The German wiki uses "Kategorie:" prefixes; the English wiki uses "Category:".
Categories you may want to try (visit them in a browser to confirm they exist
and are populated):

    https://riordan.fandom.com/de/wiki/Kategorie:Monster
    https://riordan.fandom.com/de/wiki/Kategorie:Kreaturen
    https://riordan.fandom.com/de/wiki/Kategorie:Wesen
    https://riordan.fandom.com/en/wiki/Category:Monsters
    https://riordan.fandom.com/en/wiki/Category:Creatures

You can also pass --list-categories KEYWORD to search for matching categories.

Notes on copyright
------------------
Fandom text is CC-BY-SA, but uploaded images often retain the rights of their
original creators (publishers, artists, etc.). Treat the downloads as a
research / personal-use dataset; don't redistribute.
"""

import argparse
import re
import sys
import time
from pathlib import Path
from urllib.parse import unquote

import requests

USER_AGENT = "RiordanCreatureScraper/1.0 (personal ML dataset; contact: user)"
SLEEP = 0.4  # seconds between API calls — be polite


def slugify(name: str) -> str:
    """Make a page title safe to use as a folder name."""
    name = unquote(name)
    name = re.sub(r"[^\w\-. ]", "_", name)
    return name.strip().replace(" ", "_") or "untitled"


def api_get(session: requests.Session, api_url: str, params: dict) -> dict:
    """GET the MediaWiki API with simple exponential-backoff retries."""
    params = {**params, "format": "json"}
    for attempt in range(3):
        try:
            r = session.get(api_url, params=params, timeout=30)
            r.raise_for_status()
            return r.json()
        except requests.RequestException as e:
            if attempt == 2:
                raise
            print(f"  retry {attempt + 1}: {e}", file=sys.stderr)
            time.sleep(2 ** attempt)


def iter_category_pages(session, api_url, category):
    """Yield titles of all pages (not subcategories or files) in a category."""
    cont = {}
    while True:
        data = api_get(session, api_url, {
            "action": "query",
            "list": "categorymembers",
            "cmtitle": category,
            "cmlimit": "500",
            "cmtype": "page",
            **cont,
        })
        for m in data["query"]["categorymembers"]:
            yield m["title"]
        if "continue" in data:
            cont = data["continue"]
            time.sleep(SLEEP)
        else:
            break


def get_page_image_titles(session, api_url, page_title):
    """Return the File:... titles of every image embedded on a page."""
    titles, cont = [], {}
    while True:
        data = api_get(session, api_url, {
            "action": "query",
            "prop": "images",
            "titles": page_title,
            "imlimit": "500",
            **cont,
        })
        for page in data["query"]["pages"].values():
            for img in page.get("images", []):
                titles.append(img["title"])
        if "continue" in data:
            cont = data["continue"]
            time.sleep(SLEEP)
        else:
            break
    return titles


def resolve_image_urls(session, api_url, file_titles):
    """Map File:... titles to a dict of imageinfo (url, mime, width, height)."""
    out = {}
    for i in range(0, len(file_titles), 50):  # API allows 50 titles per call
        batch = file_titles[i:i + 50]
        data = api_get(session, api_url, {
            "action": "query",
            "prop": "imageinfo",
            "iiprop": "url|mime|size",
            "titles": "|".join(batch),
        })
        for page in data["query"]["pages"].values():
            info = page.get("imageinfo")
            if info:
                out[page["title"]] = info[0]
        time.sleep(SLEEP)
    return out


def is_useful_image(info: dict) -> bool:
    """Filter out icons, SVGs, and tiny placeholders."""
    mime = info.get("mime", "")
    if not mime.startswith("image/") or mime == "image/svg+xml":
        return False
    if info.get("width", 0) < 120 or info.get("height", 0) < 120:
        return False
    return True


def download(session, url, dest: Path) -> bool:
    """Stream a file to disk. Returns True if newly downloaded, False if skipped."""
    if dest.exists():
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    with session.get(url, timeout=60, stream=True) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    return True


def list_categories(session, api_url, keyword):
    """Print categories whose title contains `keyword` (case-insensitive)."""
    cont = {}
    print(f"Searching categories for '{keyword}'…")
    while True:
        data = api_get(session, api_url, {
            "action": "query",
            "list": "allcategories",
            "acprefix": "",
            "aclimit": "500",
            **cont,
        })
        for c in data["query"]["allcategories"]:
            name = c["*"]
            if keyword.lower() in name.lower():
                print(f"  • Kategorie:{name}" if "/de/" in api_url else f"  • Category:{name}")
        if "continue" in data:
            cont = data["continue"]
            time.sleep(SLEEP)
        else:
            break


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--lang", default="de", help="Wiki language subdomain (default: de)")
    p.add_argument("--category", default="Kategorie:Monster",
                   help='Category to scrape (default: "Kategorie:Monster")')
    p.add_argument("--output", default="riordan_creatures",
                   help="Output directory (default: ./riordan_creatures)")
    p.add_argument("--limit", type=int, default=None,
                   help="Cap on number of pages (useful for testing)")
    p.add_argument("--list-categories", metavar="KEYWORD",
                   help="Don't scrape — just list categories matching KEYWORD")
    args = p.parse_args()

    api_url = f"https://riordan.fandom.com/{args.lang}/api.php"
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    if args.list_categories:
        list_categories(session, api_url, args.list_categories)
        return

    out_dir = Path(args.output)
    out_dir.mkdir(exist_ok=True)

    print(f"API: {api_url}")
    print(f"Category: {args.category}")
    print(f"Output: {out_dir.resolve()}\n")

    pages = list(iter_category_pages(session, api_url, args.category))
    if not pages:
        print("No pages found. Double-check the category name (try --list-categories).")
        return
    if args.limit:
        pages = pages[:args.limit]
    print(f"Found {len(pages)} pages.\n")

    n_new = n_skip = n_fail = 0
    for i, title in enumerate(pages, 1):
        print(f"[{i}/{len(pages)}] {title}")
        try:
            file_titles = get_page_image_titles(session, api_url, title)
        except Exception as e:
            print(f"  ! could not list images: {e}", file=sys.stderr)
            n_fail += 1
            continue
        if not file_titles:
            print("  (no images)")
            continue

        try:
            info_map = resolve_image_urls(session, api_url, file_titles)
        except Exception as e:
            print(f"  ! could not resolve URLs: {e}", file=sys.stderr)
            n_fail += 1
            continue

        folder = out_dir / slugify(title)
        for file_title, info in info_map.items():
            if not is_useful_image(info):
                continue
            url = info["url"]
            fname = unquote(url.rsplit("/", 1)[-1].split("?")[0])
            dest = folder / fname
            try:
                if download(session, url, dest):
                    n_new += 1
                    print(f"  ✓ {fname}")
                else:
                    n_skip += 1
            except Exception as e:
                n_fail += 1
                print(f"  ✗ {fname}: {e}", file=sys.stderr)
            time.sleep(SLEEP)

    print(f"\nDone. {n_new} new, {n_skip} already present, {n_fail} failed.")


if __name__ == "__main__":
    main()