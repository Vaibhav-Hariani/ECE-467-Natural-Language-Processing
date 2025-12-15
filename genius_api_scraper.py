#!/usr/bin/env python3
"""
Genius API scraper using official API with access token.
Fetch song lyrics, metadata, and save to SQLite database.
"""
import sqlite3
import sys
import re

import requests
from bs4 import BeautifulSoup


# Configuration - Modify these values as needed
GENIUS_CLIENT_ID = "YOUR_CLIENT_ID_HERE"
GENIUS_CLIENT_SECRET = "YOUR_CLIENT_SECRET_HERE"
ARTIST_NAME = "Drake"
SONG_TITLE = "Hotline Bling"
DATABASE_PATH = "genius_songs.db"


class GeniusAPI:
    def __init__(self, client_id=None, client_secret=None, client_access_token=None):
        """
        Initialize Genius API client.
        For most cases, just use the client_access_token (simplest approach).
        OAuth2 flow with client_id and client_secret is for more complex auth scenarios.
        """
        self.base_url = "https://api.genius.com"
        
        # If client_access_token is provided, use it directly (recommended)
        if client_access_token:
            self.access_token = client_access_token
            self.headers = {"Authorization": f"Bearer {client_access_token}"}
        # Otherwise, use OAuth2 client credentials flow (if needed)
        elif client_id and client_secret:
            self.access_token = self._get_oauth_token(client_id, client_secret)
            self.headers = {"Authorization": f"Bearer {self.access_token}"}
        else:
            raise ValueError("Must provide either client_access_token or both client_id and client_secret")
    
    def _get_oauth_token(self, client_id, client_secret):
        """Get OAuth2 token using client credentials flow."""
        token_url = "https://api.genius.com/oauth/token"
        data = {
            "client_id": client_id,
            "client_secret": client_secret,
            "grant_type": "client_credentials"
        }
        
        response = requests.post(token_url, data=data, timeout=15)
        response.raise_for_status()
        token_data = response.json()
        return token_data.get("access_token")
    
    def search_song(self, artist, song_title):
        """Search for a song by artist and title."""
        query = f"{artist} {song_title}"
        endpoint = f"{self.base_url}/search"
        params = {"q": query}
        
        print(f"[INFO] Searching for '{artist} - {song_title}'...")
        
        response = requests.get(endpoint, headers=self.headers, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()
        
        if data.get("response") and data["response"].get("hits"):
            for hit in data["response"]["hits"]:
                result = hit.get("result", {})
                result_title = result.get("title", "").lower()
                result_artist = result.get("primary_artist", {}).get("name", "").lower()
                
                # Check if it's a close match
                if song_title.lower() in result_title or result_title in song_title.lower():
                    if artist.lower() in result_artist or result_artist in artist.lower():
                        print(f"[FOUND] {result.get('title')} by {result.get('primary_artist', {}).get('name')}")
                        return result
            
            # If no exact match, return first result
            print("[WARN] No exact match found, using first result")
            first_result = data["response"]["hits"][0]["result"]
            print(f"[FOUND] {first_result.get('title')} by {first_result.get('primary_artist', {}).get('name')}")
            return first_result
        else:
            print("[ERROR] No results found")
            return None
    
    def get_song_details(self, song_id):
        """Get detailed information about a song."""
        endpoint = f"{self.base_url}/songs/{song_id}"
        
        response = requests.get(endpoint, headers=self.headers, timeout=15)
        response.raise_for_status()
        data = response.json()
        return data.get("response", {}).get("song", {})
    
    def get_lyrics(self, song_url):
        """Scrape lyrics from the song page URL."""
        print(f"[INFO] Fetching lyrics from {song_url}")
        
        response = requests.get(song_url, timeout=15)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "lxml")
        
        # Find lyrics containers
        lyric_divs = soup.find_all(attrs={"data-lyrics-container": True})
        if lyric_divs:
            parts = []
            for div in lyric_divs:
                text = div.get_text(separator="\n", strip=True)
                
                # Skip translation sections
                lines = text.split("\n")
                if lines and re.match(r"\[.*?[Tt]ranslation.*?\]", lines[0].strip()):
                    continue
                
                # Remove everything before and including the first [] section
                text = re.sub(r"^.*?\[\w+.*?\]\s*", "", text, flags=re.DOTALL)
                
                if text.strip():
                    parts.append(text)
            
            return "\n\n".join(parts).strip()
        else:
            # Fallback for old layout
            div = soup.find("div", class_=re.compile(r"lyrics", re.I))
            if div:
                text = div.get_text(separator="\n", strip=True)
                text = re.sub(r"^.*?\[\w+.*?\]\s*", "", text, flags=re.DOTALL)
                return text
        
        return ""


def save_to_database(db_path, song_data):
    """Save song data to SQLite database."""
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
    
    cursor.execute("""
        INSERT INTO songs (title, artists, producers, url, lyrics, annotations)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        song_data.get("title"),
        song_data.get("artists"),
        song_data.get("producers"),
        song_data.get("url"),
        song_data.get("lyrics"),
        song_data.get("annotations")
    ))
    conn.commit()
    print(f"[SAVED] {song_data.get('title')} by {song_data.get('artists')}")
    conn.close()


# Configuration - Modify these values as needed
GENIUS_API_TOKEN = "YOUR_API_TOKEN_HERE"
ARTIST_NAME = "Drake"
SONG_TITLE = "Hotline Bling"
DATABASE_PATH = "genius_songs.db"


def main():
    # Initialize API client (uses client_access_token by default)
    genius = GeniusAPI(client_access_token=GENIUS_CLIENT_ACCESS_TOKEN)
    
    # Alternative: Use OAuth2 client credentials flow
    # genius = GeniusAPI(client_id=GENIUS_CLIENT_ID, client_secret=GENIUS_CLIENT_SECRET)
    
    # Search for the song
    song_result = genius.search_song(ARTIST_NAME, SONG_TITLE)
    if not song_result:
        print("[ERROR] Song not found")
        sys.exit(1)
    
    # Get detailed song information
    song_id = song_result.get("id")
    song_details = genius.get_song_details(song_id)
    
    # Get lyrics from the song page
    song_url = song_result.get("url")
    lyrics = genius.get_lyrics(song_url)
    
    # Format artists and producers as semicolon-separated strings
    artists = song_details.get("primary_artist", {}).get("name", "")
    producers = []  # Genius API doesn't provide producer info easily, would need additional scraping
    
    # Get annotations (empty for now, would need additional API calls)
    annotations = ""
    
    # Prepare data for database
    song_data = {
        "title": song_details.get("title"),
        "artists": artists,
        "producers": ";".join(producers) if producers else "",
        "url": song_url,
        "lyrics": lyrics,
        "annotations": annotations
    }
    
    # Save to database
    save_to_database(DATABASE_PATH, song_data)
    
    print(f"\n[SUCCESS] Song data saved to {DATABASE_PATH}")
    print(f"Title: {song_data['title']}")
    print(f"Artists: {song_data['artists']}")
    print(f"Producers: {song_data['producers']}")
    print(f"Lyrics length: {len(lyrics)} characters")


if __name__ == "__main__":
    main()
