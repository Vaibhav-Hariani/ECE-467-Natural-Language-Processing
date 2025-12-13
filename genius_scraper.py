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
    data = {"url": url, "title": None, "artists": [], "producers": [], "lyrics": "", "annotations": [], "comments": []}

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
            parts.append(d.get_text(separator="\n", strip=True))
        data["lyrics"] = "\n\n".join(parts).strip()
    else:
        # old layout
        div = soup.find("div", class_=re.compile(r"lyrics", re.I))
        if div:
            data["lyrics"] = div.get_text(separator="\n", strip=True)

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

    # Comments: try to find comment elements and filter last-day ones
    comments = []
    # look for common comment containers
    possible = []
    for c in soup.find_all(class_=re.compile(r"comment", re.I)):
        possible.append(c)

    # Heuristic: for each comment-like block, find text and time text
    for c in possible:
        text = c.get_text(separator=" \n", strip=True)
        # find time-like strings
        time_txt = None
        time_tag = c.find(class_=re.compile(r"time|date|timestamp", re.I))
        if time_tag:
            time_txt = time_tag.get_text(strip=True)
        else:
            # search for 'ago' tokens in text
            m = re.search(r"(\d+\s+(?:minute|minutes|hour|hours|day|days)\s+ago)", text, re.I)
            if m:
                time_txt = m.group(1)

        # determine if within last day
        within_day = False
        if time_txt:
            if re.search(r"minute|hour", time_txt, re.I):
                within_day = True
            elif re.search(r"day", time_txt, re.I):
                # allow '1 day ago'
                if re.search(r"1\s+day", time_txt, re.I):
                    within_day = True
        else:
            # if no time found, still include as we can't be sure
            within_day = True

        if within_day:
            # extract short text
            snippet = text
            # keep reasonable length
            if len(snippet) > 1000:
                snippet = snippet[:1000] + "..."
            comments.append(snippet)
        if len(comments) >= 10:
            break

    data["comments"] = comments[:10]

    return data


def crawl(seed_urls, limit=1000, delay=1.5, session=None, max_pages=200):
    session = session or requests.Session()
    collected = []
    seen = set()
    to_visit = list(seed_urls)
    page_count = 0

    while to_visit and len(collected) < limit and page_count < max_pages:
        page = to_visit.pop(0)
        page_count += 1
        print(f"[INFO] Visiting seed/list page: {page} (page {page_count})")
        html = fetch(page, session=session)
        if not html:
            time.sleep(delay)
            continue

        song_links = find_song_links_from_page(html, base_url=page)
        for link in song_links:
            if link in seen:
                continue
            seen.add(link)
            collected.append(link)
            if len(collected) >= limit:
                break

        # find next-page links to continue discovering songs
        soup = BeautifulSoup(html, "lxml")
        next_link = None
        for a in soup.find_all("a", href=True):
            if re.search(r"page=\d+|/page/\d+", a["href"]):
                href = a["href"]
                next_link = urljoin(page, href)
                break
        if next_link and next_link not in to_visit:
            to_visit.append(next_link)

        time.sleep(delay)

    return collected[:limit]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", "-s", nargs="*", default=["https://genius.com/"], help="Seed pages to start discovery from")
    p.add_argument("--limit", "-n", type=int, default=100, help="How many songs to collect (max 1000 suggested)")
    p.add_argument("--delay", "-d", type=float, default=1.5, help="Seconds to wait between requests")
    p.add_argument("--verbose", "-v", action="store_true")
    p.add_argument("--out", "-o", default="genius_songs.csv", help="CSV output file to write results into")
    args = p.parse_args()

    session = requests.Session()

    print(f"[INFO] Starting discovery from seeds: {args.seeds}")
    song_urls = crawl(args.seeds, limit=args.limit, delay=args.delay, session=session)
    print(f"[INFO] Discovered {len(song_urls)} song pages (capped by --limit).\n")

    # Open CSV for incremental writing
    out_path = args.out
    with open(out_path, "w", newline="", encoding="utf-8") as csvf:
        writer = csv.DictWriter(csvf, fieldnames=["title", "artists", "producers", "url", "lyrics", "annotations", "comments"])
        writer.writeheader()

        for idx, url in enumerate(song_urls, start=1):
            print(f"[INFO] Processing {idx}/{len(song_urls)}: {url}")
            html = fetch(url, session=session)
            if not html:
                continue
            info = parse_song_page(html, url)

            title = info.get("title") or ""
            artists = "; ".join(info.get("artists") or [])
            producers = "; ".join(info.get("producers") or [])

            # prepare lyrics, annotations, comments for CSV
            lyrics = info.get("lyrics") or ""
            # join annotations and comments using a separator; keep their whitespace collapsed
            annotations = info.get("annotations") or []
            annotations = [re.sub(r"\s+", " ", a).strip() for a in annotations]
            annotations_text = " || ".join(annotations)

            comments = info.get("comments") or []
            comments = [re.sub(r"\s+", " ", c).strip() for c in comments]
            comments_text = " || ".join(comments)

            writer.writerow({
                "title": title,
                "artists": artists,
                "producers": producers,
                "url": url,
                "lyrics": lyrics,
                "annotations": annotations_text,
                "comments": comments_text,
            })

            # feedback
            print(f"[SAVED] {title} — artists: {artists or '(none)'} — producers: {producers or '(none)'} — annotations: {len(annotations)} — comments: {len(comments)}")

            # polite pause between song pages
            time.sleep(args.delay)


if __name__ == "__main__":
    main()
