#!/usr/bin/env python3
"""
Best-effort Genius scraper (page-by-page, no API).

Features (heuristic):
- Crawl seed pages and collect song URLs (links ending with "-lyrics").
- For each song page, parse and print: title, artists, producers, lyrics, annotations, and up-to-10 recent comments (last day heuristic).
- Respects a polite delay between requests.

Notes:
- This is a pragmatic, best-effort scraper. Genius pages change often; selectors are heuristic and may need tuning.
- Do not run too fast; use --delay to increase politeness.
"""
import argparse
import time
import re
import sys
from urllib.parse import urljoin, urlparse
import csv
import sqlite3

import requests
from bs4 import BeautifulSoup

# # User-agent to identify ourselves politely
# HEADERS = {
#     "User-Agent": "GeniusScraper/0.1 (+https://github.com/) - polite crawler for research"
# }


def fetch(url, session=None, timeout=15):
    s = session or requests.Session()
    try:
        resp = s.get(url, timeout=timeout)
        resp.raise_for_status()
        return resp.text
    except Exception as e:
        print(f"[ERROR] Fetch failed for {url}: {e}", file=sys.stderr)
        return None


def find_song_links_from_page(html, base_url="https://genius.com"):
    """Return absolute URLs that look like song lyric pages ending with "-lyrics"."""
    soup = BeautifulSoup(html, "lxml")
    links = set()
    for a in soup.find_all("a", href=True):
        href = a["href"]
        # canonicalize
        parsed = urlparse(href)
        if not parsed.netloc:
            href = urljoin(base_url, href)
        if re.search(r"/[-A-Za-z0-9\.%_]+-lyrics$", href):
            links.add(href.split("?")[0])
    return list(links)


def parse_song_page(html, url):
    soup = BeautifulSoup(html, "lxml")
    data = {"url": url, "title": None, "artists": [], "producers": [], "lyrics": "", "annotations": []}

    # Title
    og = soup.find("meta", property="og:title")
    if og and og.get("content"):
        data["title"] = og["content"].strip()
    else:
        h1 = soup.find("h1")
        if h1:
            data["title"] = h1.get_text(strip=True)

    # Artists: robust extraction
    def clean_name(n: str):
        if not n:
            return None
        s = re.sub(r"\s+\(.*?\)$", "", n).strip()
        s = re.sub(r"\s+\[.*?\]$", "", s).strip()
        s = re.sub(r"\s+Verified Artists$", "", s).strip()
        # ignore obvious JS/analytics junk
        if re.search(r"\b_qevents\b|qacct|push\(|\{|\}|<script|http", s, re.I):
            return None
        # short garbage
        if len(re.sub(r"[^A-Za-z]", "", s)) < 2:
            return None
        return s

    artists = []
    # 1) meta author
    meta_author = soup.find("meta", attrs={"name": "author"})
    if meta_author and meta_author.get("content"):
        a = clean_name(meta_author["content"].strip())
        if a:
            artists.append(a)

    # 2) itemprop byArtist / rel=author
    for tag in soup.find_all(attrs={"itemprop": "byArtist"}):
        for a in tag.find_all("a", href=True):
            name = clean_name(a.get_text(strip=True))
            if name and name not in artists:
                artists.append(name)

    for a in soup.find_all("a", href=True):
        href = a.get("href") or ""
        # Prefer links that look like artist pages (contain '/artists' or '/artist')
        if re.search(r"/artists?/", href):
            name = clean_name(a.get_text(strip=True))
            if name and name not in artists:
                artists.append(name)

    # fallback: parse from og:title or page title
    if not artists and data.get("title"):
        og = data.get("title")
        parts = re.split(r"\s+[-–—]\s+", og)
        if parts:
            name = clean_name(parts[0])
            if name:
                artists.append(name)

    data["artists"] = artists

    # Lyrics: modern Genius uses data-lyrics-container attributes
    lyric_divs = soup.find_all(attrs={"data-lyrics-container": True})
    if lyric_divs:
        parts = []
        for d in lyric_divs:
            # Get the text content
            text = d.get_text(separator="\n", strip=True)
            
            # Skip translation sections (they start with [Translation] or similar)
            # Only skip if the ENTIRE block is a translation section
            lines = text.split("\n")
            if lines and re.match(r"\[.*?[Tt]ranslation.*?\]", lines[0].strip()):
                continue
            
            # Remove everything before and including the first [] section
            text = re.sub(r"^.*?\[\w+.*?\]\s*", "", text, flags=re.DOTALL)
            
            if text.strip():
                parts.append(text)
        
        data["lyrics"] = "\n\n".join(parts).strip()
    else:
        # old layout
        div = soup.find("div", class_=re.compile(r"lyrics", re.I))
        if div:
            text = div.get_text(separator="\n", strip=True)
            # Remove everything before and including the first [] section
            text = re.sub(r"^.*?\[\w+.*?\]\s*", "", text, flags=re.DOTALL)
            data["lyrics"] = text

    # Annotations: attempt to find annotation blocks
    annos = []
    # look for elements with 'annotation' in class
    for tag in soup.find_all(class_=re.compile(r"annotation", re.I)):
        txt = tag.get_text(separator=" ", strip=True)
        if txt:
            annos.append(txt)
    # also look for 'referent' blocks or 'lyrics' commentary
    for tag in soup.find_all(class_=re.compile(r"referent|annotation_body|annotation_text", re.I)):
        txt = tag.get_text(separator=" ", strip=True)
        if txt and txt not in annos:
            annos.append(txt)
    data["annotations"] = annos

    # Producers: safer extraction avoiding script/text nodes
    def plausible_name(s: str):
        if not s:
            return None
        s = re.sub(r"\s+\(.*?\)$", "", s).strip()
        s = re.sub(r"\s+\[.*?\]$", "", s).strip()
        if re.search(r"\b_qevents\b|qacct|push\(|\{|\}|<script|http", s, re.I):
            return None
        if len(re.sub(r"[^A-Za-z]", "", s)) < 2:
            return None
        return s

    producers = []
    # search visible container tags likely to hold credits
    for tag in soup.find_all(["li", "dd", "div", "p", "span"]):
        text = tag.get_text(separator=" ", strip=True)
        if not text:
            continue
        if re.search(r"Produced by|Produced:\s*|Producer:\s*", text, re.I):
            # prefer links inside this tag
            for a in tag.find_all("a", href=True):
                name = plausible_name(a.get_text(strip=True))
                if name and name not in producers:
                    producers.append(name)
            # also parse trailing text after 'Produced by'
            parts = re.split(r"Produced by[:\s]+", text, flags=re.I)
            if len(parts) > 1:
                tail = parts[1]
                for p in re.split(r",|;| and | & |\band\b", tail):
                    name = plausible_name(p.strip())
                    if name and name not in producers:
                        producers.append(name)

    # look for dedicated credits sections
    for credit_block in soup.find_all(class_=re.compile(r"credit|credits|SongCredits|SongMeta", re.I)):
        txt = credit_block.get_text(separator=" ", strip=True)
        if re.search(r"Produced by|Producer", txt, re.I):
            for a in credit_block.find_all("a", href=True):
                name = plausible_name(a.get_text(strip=True))
                if name and name not in producers:
                    producers.append(name)
            parts = re.split(r"Produced by[:\s]+", txt, flags=re.I)
            if len(parts) > 1:
                tail = parts[1]
                for p in re.split(r",|;| and | & |\band\b", tail):
                    name = plausible_name(p.strip())
                    if name and name not in producers:
                        producers.append(name)

    data["producers"] = producers

    return data


def crawl(seed_urls, limit=1000, delay=0.5, session=None, max_pages=5000):
    session = session or requests.Session()
    collected = []
    seen = set()
    to_visit = list(seed_urls)
    visited_pages = set()
    page_count = 0

    while to_visit and len(collected) < limit and page_count < max_pages:
        page = to_visit.pop(0)
        if page in visited_pages:
            continue
        visited_pages.add(page)
        page_count += 1
        print(f"[INFO] Visiting seed/list page: {page} (page {page_count}, collected {len(collected)} songs)")
        html = fetch(page, session=session)
        if not html:
            time.sleep(delay)
            continue

        song_links = find_song_links_from_page(html, base_url=page)
        print(f"[DEBUG] Found {len(song_links)} song links on this page")
        for link in song_links:
            if link in seen:
                continue
            seen.add(link)
            collected.append(link)
            if len(collected) >= limit:
                break

        # find next-page links to continue discovering songs
        soup = BeautifulSoup(html, "lxml")
        next_links = []
        for a in soup.find_all("a", href=True):
            href = a.get("href") or ""
            # Look for pagination links and artist pages
            if re.search(r"page=\d+|/page/\d+|/artists-index/", href):
                next_url = urljoin(page, href)
                if next_url not in visited_pages and next_url not in to_visit:
                    next_links.append(next_url)
        
        print(f"[DEBUG] Found {len(next_links)} new pages to visit")
        # Add found next-page links to queue
        for next_link in next_links:
            to_visit.append(next_link)

        time.sleep(delay)

    print(f"[INFO] Finished crawl: visited {page_count} pages, collected {len(collected)} songs")
    return collected[:limit]


def search_song(artist, song_title, session=None):
    """Search for a specific song on Genius and return its URL."""
    session = session or requests.Session()
    
    # Format the search query
    query = f"{artist} {song_title}".replace(" ", "+")
    search_url = f"https://genius.com/api/search/multi?q={query}"
    
    print(f"[INFO] Searching for '{artist} - {song_title}'...")
    
    try:
        # Try API search first
        resp = session.get(search_url, timeout=15)
        if resp.status_code == 200:
            data = resp.json()
            if "response" in data and "sections" in data["response"]:
                for section in data["response"]["sections"]:
                    if section.get("type") == "song" and "hits" in section:
                        for hit in section["hits"]:
                            if "result" in hit:
                                result = hit["result"]
                                result_title = result.get("title", "").lower()
                                result_artist = result.get("primary_artist", {}).get("name", "").lower()
                                
                                # Check if it's a match
                                if song_title.lower() in result_title or result_title in song_title.lower():
                                    if artist.lower() in result_artist or result_artist in artist.lower():
                                        url = result.get("url")
                                        if url:
                                            print(f"[FOUND] {result.get('title')} by {result.get('primary_artist', {}).get('name')}")
                                            return url
    except Exception as e:
        print(f"[WARN] API search failed: {e}, trying web search...")
    
    # Fallback: try constructing URL directly
    # Genius URLs follow pattern: https://genius.com/Artist-song-title-lyrics
    artist_slug = re.sub(r'[^a-zA-Z0-9]+', '-', artist.lower()).strip('-')
    song_slug = re.sub(r'[^a-zA-Z0-9]+', '-', song_title.lower()).strip('-')
    constructed_url = f"https://genius.com/{artist_slug}-{song_slug}-lyrics"
    
    print(f"[INFO] Trying constructed URL: {constructed_url}")
    html = fetch(constructed_url, session=session)
    if html and "404" not in html[:1000]:
        print(f"[FOUND] Song found at constructed URL")
        return constructed_url
    
    print(f"[ERROR] Could not find song '{artist} - {song_title}'")
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--artist", "-a", help="Artist name to search for")
    p.add_argument("--song", "-t", help="Song title to search for")
    p.add_argument("--delay", "-d", type=float, default=0.5, help="Seconds to wait between requests")
    p.add_argument("--verbose", "-v", action="store_true")
    p.add_argument("--db", "-b", default="genius_songs.db", help="SQLite database file to write results into")
    args = p.parse_args()

    session = requests.Session()

    # Check if artist and song are provided
    if not args.artist or not args.song:
        print("[ERROR] Please provide both --artist and --song arguments")
        print("Example: python genius_scraper.py --artist 'Drake' --song 'Hotline Bling'")
        sys.exit(1)

    # Search for the specific song
    song_url = search_song(args.artist, args.song, session=session)
    
    if not song_url:
        print("[ERROR] Song not found")
        sys.exit(1)
    
    song_urls = [song_url]
    print(f"[INFO] Found song URL: {song_url}\n")

    # Initialize SQLite database
    db_path = args.db
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS songs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT,
            artists TEXT,
            producers TEXT,
            url TEXT UNIQUE,
            lyrics TEXT,
            annotations TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()

    for idx, url in enumerate(song_urls, start=1):
        print(f"[INFO] Processing {idx}/{len(song_urls)}: {url}")
        html = fetch(url, session=session)
        if not html:
            continue
        info = parse_song_page(html, url)

        title = info.get("title") or ""
        artists = "; ".join(info.get("artists") or [])
        producers = "; ".join(info.get("producers") or [])

        # prepare lyrics and annotations for database
        lyrics = info.get("lyrics") or ""
        # join annotations using a separator; keep their whitespace collapsed
        annotations = info.get("annotations") or []
        annotations = [re.sub(r"\s+", " ", a).strip() for a in annotations]
        annotations_text = " || ".join(annotations)

        try:
            cursor.execute("""
                INSERT INTO songs (title, artists, producers, url, lyrics, annotations)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (title, artists, producers, url, lyrics, annotations_text))
            conn.commit()
            
            # feedback
            print(f"[SAVED] {title} — artists: {artists or '(none)'} — producers: {producers or '(none)'} — annotations: {len(annotations)}")
        except sqlite3.IntegrityError:
            print(f"[SKIP] {title} already exists in database")

        # polite pause between song pages
        time.sleep(0.5)

    conn.close()
    print(f"\n[INFO] Data saved to {db_path}")


if __name__ == "__main__":
    main()
