"""
RateYourMusic Scraper
Scrapes top songs/tracks from RateYourMusic and stores them in a SQLite database.
Includes genres, descriptors, artists, albums, and top 100 reviews per song.
"""

import sqlite3
import time
import random
import json
import re
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Any
from dataclasses import dataclass, field, asdict

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import (
    TimeoutException, 
    NoSuchElementException,
    StaleElementReferenceException,
    WebDriverException
)
from bs4 import BeautifulSoup

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('scraper.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class Artist:
    name: str
    rym_url: str
    role: str = "primary"  # primary, featuring, producer, etc.
    
@dataclass
class Review:
    username: str
    rating: Optional[float]
    review_text: str
    date: str
    helpful_count: int = 0

@dataclass
class Track:
    title: str
    rym_url: str
    album_title: Optional[str] = None
    album_url: Optional[str] = None
    release_year: Optional[int] = None
    average_rating: Optional[float] = None
    num_ratings: Optional[int] = None
    genres: List[str] = field(default_factory=list)
    descriptors: List[str] = field(default_factory=list)
    artists: List[Artist] = field(default_factory=list)
    reviews: List[Review] = field(default_factory=list)
    duration: Optional[str] = None
    chart_position: Optional[int] = None

# ============================================================================
# DATABASE SETUP
# ============================================================================

def create_database(db_path: str = "rym_music.db") -> sqlite3.Connection:
    """Create SQLite database with proper schema for storing music data."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Enable foreign keys
    cursor.execute("PRAGMA foreign_keys = ON")
    
    # Artists table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS artists (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            rym_url TEXT UNIQUE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Albums table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS albums (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            rym_url TEXT UNIQUE,
            release_year INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Tracks table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS tracks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            rym_url TEXT UNIQUE,
            album_id INTEGER,
            release_year INTEGER,
            average_rating REAL,
            num_ratings INTEGER,
            duration TEXT,
            chart_position INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (album_id) REFERENCES albums(id)
        )
    """)
    
    # Genres table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS genres (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL
        )
    """)
    
    # Descriptors table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS descriptors (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL
        )
    """)
    
    # Track-Artist junction table (many-to-many)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS track_artists (
            track_id INTEGER,
            artist_id INTEGER,
            role TEXT DEFAULT 'primary',
            PRIMARY KEY (track_id, artist_id, role),
            FOREIGN KEY (track_id) REFERENCES tracks(id),
            FOREIGN KEY (artist_id) REFERENCES artists(id)
        )
    """)
    
    # Album-Artist junction table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS album_artists (
            album_id INTEGER,
            artist_id INTEGER,
            role TEXT DEFAULT 'primary',
            PRIMARY KEY (album_id, artist_id, role),
            FOREIGN KEY (album_id) REFERENCES albums(id),
            FOREIGN KEY (artist_id) REFERENCES artists(id)
        )
    """)
    
    # Track-Genre junction table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS track_genres (
            track_id INTEGER,
            genre_id INTEGER,
            PRIMARY KEY (track_id, genre_id),
            FOREIGN KEY (track_id) REFERENCES tracks(id),
            FOREIGN KEY (genre_id) REFERENCES genres(id)
        )
    """)
    
    # Track-Descriptor junction table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS track_descriptors (
            track_id INTEGER,
            descriptor_id INTEGER,
            PRIMARY KEY (track_id, descriptor_id),
            FOREIGN KEY (track_id) REFERENCES tracks(id),
            FOREIGN KEY (descriptor_id) REFERENCES descriptors(id)
        )
    """)
    
    # Reviews table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS reviews (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            track_id INTEGER,
            username TEXT,
            rating REAL,
            review_text TEXT,
            review_date TEXT,
            helpful_count INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (track_id) REFERENCES tracks(id),
            UNIQUE(track_id, username, review_date)
        )
    """)
    
    # Scrape progress table (for resuming interrupted scrapes)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS scrape_progress (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            last_page INTEGER DEFAULT 0,
            last_track_url TEXT,
            total_tracks_scraped INTEGER DEFAULT 0,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    conn.commit()
    logger.info(f"Database created/connected: {db_path}")
    return conn

# ============================================================================
# DATABASE OPERATIONS
# ============================================================================

class DatabaseManager:
    """Handles all database operations for the scraper."""
    
    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn
        self.cursor = conn.cursor()
    
    def get_or_create_artist(self, name: str, rym_url: str = None) -> int:
        """Get existing artist ID or create new artist."""
        if rym_url:
            self.cursor.execute("SELECT id FROM artists WHERE rym_url = ?", (rym_url,))
        else:
            self.cursor.execute("SELECT id FROM artists WHERE name = ?", (name,))
        
        result = self.cursor.fetchone()
        if result:
            return result[0]
        
        self.cursor.execute(
            "INSERT INTO artists (name, rym_url) VALUES (?, ?)",
            (name, rym_url)
        )
        self.conn.commit()
        return self.cursor.lastrowid
    
    def get_or_create_album(self, title: str, rym_url: str = None, year: int = None) -> int:
        """Get existing album ID or create new album."""
        if rym_url:
            self.cursor.execute("SELECT id FROM albums WHERE rym_url = ?", (rym_url,))
            result = self.cursor.fetchone()
            if result:
                return result[0]
        
        self.cursor.execute(
            "INSERT OR IGNORE INTO albums (title, rym_url, release_year) VALUES (?, ?, ?)",
            (title, rym_url, year)
        )
        self.conn.commit()
        
        if rym_url:
            self.cursor.execute("SELECT id FROM albums WHERE rym_url = ?", (rym_url,))
        else:
            self.cursor.execute("SELECT id FROM albums WHERE title = ?", (title,))
        return self.cursor.fetchone()[0]
    
    def get_or_create_genre(self, name: str) -> int:
        """Get existing genre ID or create new genre."""
        self.cursor.execute("SELECT id FROM genres WHERE name = ?", (name,))
        result = self.cursor.fetchone()
        if result:
            return result[0]
        
        self.cursor.execute("INSERT INTO genres (name) VALUES (?)", (name,))
        self.conn.commit()
        return self.cursor.lastrowid
    
    def get_or_create_descriptor(self, name: str) -> int:
        """Get existing descriptor ID or create new descriptor."""
        self.cursor.execute("SELECT id FROM descriptors WHERE name = ?", (name,))
        result = self.cursor.fetchone()
        if result:
            return result[0]
        
        self.cursor.execute("INSERT INTO descriptors (name) VALUES (?)", (name,))
        self.conn.commit()
        return self.cursor.lastrowid
    
    def insert_track(self, track: Track) -> int:
        """Insert a track and all related data into the database."""
        # Handle album
        album_id = None
        if track.album_title:
            album_id = self.get_or_create_album(
                track.album_title, 
                track.album_url, 
                track.release_year
            )
        
        # Insert track
        self.cursor.execute("""
            INSERT OR REPLACE INTO tracks 
            (title, rym_url, album_id, release_year, average_rating, num_ratings, duration, chart_position)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            track.title,
            track.rym_url,
            album_id,
            track.release_year,
            track.average_rating,
            track.num_ratings,
            track.duration,
            track.chart_position
        ))
        self.conn.commit()
        
        # Get track ID
        self.cursor.execute("SELECT id FROM tracks WHERE rym_url = ?", (track.rym_url,))
        track_id = self.cursor.fetchone()[0]
        
        # Insert artists
        for artist in track.artists:
            artist_id = self.get_or_create_artist(artist.name, artist.rym_url)
            self.cursor.execute("""
                INSERT OR IGNORE INTO track_artists (track_id, artist_id, role)
                VALUES (?, ?, ?)
            """, (track_id, artist_id, artist.role))
            
            # Also link artist to album
            if album_id:
                self.cursor.execute("""
                    INSERT OR IGNORE INTO album_artists (album_id, artist_id, role)
                    VALUES (?, ?, ?)
                """, (album_id, artist_id, artist.role))
        
        # Insert genres
        for genre_name in track.genres:
            genre_id = self.get_or_create_genre(genre_name)
            self.cursor.execute("""
                INSERT OR IGNORE INTO track_genres (track_id, genre_id)
                VALUES (?, ?)
            """, (track_id, genre_id))
        
        # Insert descriptors
        for desc_name in track.descriptors:
            desc_id = self.get_or_create_descriptor(desc_name)
            self.cursor.execute("""
                INSERT OR IGNORE INTO track_descriptors (track_id, descriptor_id)
                VALUES (?, ?)
            """, (track_id, desc_id))
        
        # Insert reviews
        for review in track.reviews:
            self.cursor.execute("""
                INSERT OR IGNORE INTO reviews 
                (track_id, username, rating, review_text, review_date, helpful_count)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                track_id,
                review.username,
                review.rating,
                review.review_text,
                review.date,
                review.helpful_count
            ))
        
        self.conn.commit()
        return track_id
    
    def save_progress(self, page: int, track_url: str, total_scraped: int):
        """Save scraping progress for resumption."""
        self.cursor.execute("""
            INSERT OR REPLACE INTO scrape_progress 
            (id, last_page, last_track_url, total_tracks_scraped, updated_at)
            VALUES (1, ?, ?, ?, CURRENT_TIMESTAMP)
        """, (page, track_url, total_scraped))
        self.conn.commit()
    
    def get_progress(self) -> tuple:
        """Get last scraping progress."""
        self.cursor.execute("""
            SELECT last_page, last_track_url, total_tracks_scraped 
            FROM scrape_progress WHERE id = 1
        """)
        result = self.cursor.fetchone()
        return result if result else (0, None, 0)
    
    def track_exists(self, rym_url: str) -> bool:
        """Check if a track already exists in the database."""
        self.cursor.execute("SELECT 1 FROM tracks WHERE rym_url = ?", (rym_url,))
        return self.cursor.fetchone() is not None

# ============================================================================
# WEB SCRAPER
# ============================================================================

class RYMScraper:
    """Scraper for RateYourMusic using Selenium with anti-detection measures."""
    
    BASE_URL = "https://rateyourmusic.com"
    CHART_URL = f"{BASE_URL}/charts/top/track/all-time/"
    
    def __init__(self, db_manager: DatabaseManager, headless: bool = False):
        self.db = db_manager
        self.driver = None
        self.headless = headless
        self.wait_time = (3, 7)  # Random wait range in seconds
        
    def _create_driver(self) -> webdriver.Chrome:
        """Create a Chrome WebDriver with anti-detection options."""
        options = Options()
        
        if self.headless:
            options.add_argument("--headless=new")
        
        # Anti-detection measures
        options.add_argument("--disable-blink-features=AutomationControlled")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-gpu")
        options.add_argument("--window-size=1920,1080")
        options.add_argument("--start-maximized")
        
        # Realistic user agent
        options.add_argument(
            "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
        
        # Disable automation flags
        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        options.add_experimental_option("useAutomationExtension", False)
        
        driver = webdriver.Chrome(options=options)
        
        # Remove webdriver property
        driver.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument", {
            "source": """
                Object.defineProperty(navigator, 'webdriver', {
                    get: () => undefined
                });
            """
        })
        
        return driver
    
    def _random_delay(self, min_sec: float = None, max_sec: float = None):
        """Add a random delay to mimic human behavior."""
        min_s = min_sec or self.wait_time[0]
        max_s = max_sec or self.wait_time[1]
        delay = random.uniform(min_s, max_s)
        time.sleep(delay)
    
    def _wait_for_page_load(self, timeout: int = 30):
        """Wait for page to fully load."""
        try:
            WebDriverWait(self.driver, timeout).until(
                lambda d: d.execute_script("return document.readyState") == "complete"
            )
        except TimeoutException:
            logger.warning("Page load timeout, continuing anyway...")
    
    def _handle_cloudflare(self, max_attempts: int = 3):
        """Handle Cloudflare challenge if present."""
        for attempt in range(max_attempts):
            page_source = self.driver.page_source.lower()
            
            if "checking your browser" in page_source or "cloudflare" in page_source:
                logger.info(f"Cloudflare challenge detected, waiting... (attempt {attempt + 1})")
                time.sleep(10 + attempt * 5)  # Increasing wait time
                self._wait_for_page_load()
            else:
                return True
        
        logger.warning("Could not bypass Cloudflare after max attempts")
        return False
    
    def start(self):
        """Initialize the browser driver."""
        logger.info("Starting browser...")
        self.driver = self._create_driver()
        
    def stop(self):
        """Close the browser driver."""
        if self.driver:
            self.driver.quit()
            logger.info("Browser closed.")
    
    def get_chart_page(self, page: int = 1) -> List[Dict[str, str]]:
        """Fetch a page from the top tracks chart and extract track URLs."""
        url = f"{self.CHART_URL}{page}/"
        logger.info(f"Fetching chart page {page}: {url}")
        
        self.driver.get(url)
        self._wait_for_page_load()
        self._handle_cloudflare()
        self._random_delay(2, 4)
        
        soup = BeautifulSoup(self.driver.page_source, 'html.parser')
        tracks_info = []
        
        # Find chart entries - RYM uses various class names
        # Looking for track entries in the chart
        chart_entries = soup.select('.page_section_charts .page_charts_section_charts_item')
        
        if not chart_entries:
            # Alternative selectors
            chart_entries = soup.select('[class*="chart"] [class*="item"]')
        
        if not chart_entries:
            # Try finding links to tracks directly
            track_links = soup.select('a[href*="/release/single/"]')
            for link in track_links:
                href = link.get('href', '')
                if href:
                    full_url = f"{self.BASE_URL}{href}" if href.startswith('/') else href
                    title = link.get_text(strip=True)
                    tracks_info.append({
                        'url': full_url,
                        'title': title
                    })
        else:
            for entry in chart_entries:
                # Extract track URL and basic info
                track_link = entry.select_one('a[href*="/release/"]')
                if track_link:
                    href = track_link.get('href', '')
                    full_url = f"{self.BASE_URL}{href}" if href.startswith('/') else href
                    title = track_link.get_text(strip=True)
                    
                    tracks_info.append({
                        'url': full_url,
                        'title': title
                    })
        
        logger.info(f"Found {len(tracks_info)} tracks on page {page}")
        return tracks_info
    
    def scrape_track_page(self, track_url: str, chart_position: int = None) -> Optional[Track]:
        """Scrape detailed information from a track's page."""
        logger.info(f"Scraping track: {track_url}")
        
        try:
            self.driver.get(track_url)
            self._wait_for_page_load()
            self._handle_cloudflare()
            self._random_delay()
            
            soup = BeautifulSoup(self.driver.page_source, 'html.parser')
            
            # Extract track title
            title_elem = soup.select_one('.album_title, [class*="title"] h1, .release_title')
            title = title_elem.get_text(strip=True) if title_elem else "Unknown"
            
            # Extract artist(s)
            artists = []
            artist_elems = soup.select('.album_artist a, [class*="artist"] a, .credited a')
            for artist_elem in artist_elems:
                artist_name = artist_elem.get_text(strip=True)
                artist_url = artist_elem.get('href', '')
                if artist_url and not artist_url.startswith('http'):
                    artist_url = f"{self.BASE_URL}{artist_url}"
                
                # Determine role based on context
                role = "primary"
                parent_text = artist_elem.parent.get_text() if artist_elem.parent else ""
                if "feat" in parent_text.lower() or "featuring" in parent_text.lower():
                    role = "featuring"
                elif "prod" in parent_text.lower():
                    role = "producer"
                
                artists.append(Artist(name=artist_name, rym_url=artist_url, role=role))
            
            # Extract album info
            album_elem = soup.select_one('a[href*="/release/album/"]')
            album_title = album_elem.get_text(strip=True) if album_elem else None
            album_url = None
            if album_elem:
                href = album_elem.get('href', '')
                album_url = f"{self.BASE_URL}{href}" if href.startswith('/') else href
            
            # Extract rating
            rating = None
            rating_elem = soup.select_one('.avg_rating, [class*="rating_avg"], .stat_value')
            if rating_elem:
                try:
                    rating_text = rating_elem.get_text(strip=True)
                    rating = float(re.search(r'[\d.]+', rating_text).group())
                except (AttributeError, ValueError):
                    pass
            
            # Extract number of ratings
            num_ratings = None
            ratings_count_elem = soup.select_one('.num_ratings, [class*="rating_count"]')
            if ratings_count_elem:
                try:
                    count_text = ratings_count_elem.get_text(strip=True)
                    # Remove commas and extract number
                    num_ratings = int(re.sub(r'[^\d]', '', count_text))
                except (AttributeError, ValueError):
                    pass
            
            # Extract release year
            year = None
            year_elem = soup.select_one('.issue_year, [class*="year"], time')
            if year_elem:
                try:
                    year_text = year_elem.get_text(strip=True)
                    year_match = re.search(r'\b(19|20)\d{2}\b', year_text)
                    if year_match:
                        year = int(year_match.group())
                except (AttributeError, ValueError):
                    pass
            
            # Extract genres
            genres = []
            genre_elems = soup.select('.genre a, [class*="genres"] a, a[href*="/genre/"]')
            for elem in genre_elems:
                genre_name = elem.get_text(strip=True)
                if genre_name and len(genre_name) > 1:
                    genres.append(genre_name)
            
            # Extract descriptors
            descriptors = []
            desc_elems = soup.select('.descriptor, [class*="descriptors"] a, .release_pri_descriptors a')
            for elem in desc_elems:
                desc_name = elem.get_text(strip=True)
                if desc_name and len(desc_name) > 1:
                    descriptors.append(desc_name)
            
            # Extract duration
            duration = None
            duration_elem = soup.select_one('[class*="duration"], .length')
            if duration_elem:
                duration = duration_elem.get_text(strip=True)
            
            track = Track(
                title=title,
                rym_url=track_url,
                album_title=album_title,
                album_url=album_url,
                release_year=year,
                average_rating=rating,
                num_ratings=num_ratings,
                genres=list(set(genres)),  # Remove duplicates
                descriptors=list(set(descriptors)),
                artists=artists,
                duration=duration,
                chart_position=chart_position
            )
            
            return track
            
        except Exception as e:
            logger.error(f"Error scraping track {track_url}: {e}")
            return None
    
    def scrape_track_reviews(self, track: Track, max_reviews: int = 100) -> List[Review]:
        """Scrape reviews for a track."""
        reviews = []
        
        # Construct reviews URL
        # RYM review pages typically follow pattern: /release/.../reviews/
        review_url = track.rym_url.rstrip('/') + '/reviews/'
        
        try:
            logger.info(f"Scraping reviews from: {review_url}")
            self.driver.get(review_url)
            self._wait_for_page_load()
            self._handle_cloudflare()
            self._random_delay()
            
            page = 1
            while len(reviews) < max_reviews:
                soup = BeautifulSoup(self.driver.page_source, 'html.parser')
                
                # Find review elements
                review_elems = soup.select('.review, [class*="review_item"], .user_review')
                
                if not review_elems:
                    # Try alternative selectors
                    review_elems = soup.select('[id*="review"]')
                
                if not review_elems:
                    logger.info("No reviews found on this page")
                    break
                
                for review_elem in review_elems:
                    if len(reviews) >= max_reviews:
                        break
                    
                    # Extract username
                    user_elem = review_elem.select_one('a[href*="/~"], .reviewer, .user')
                    username = user_elem.get_text(strip=True) if user_elem else "Anonymous"
                    
                    # Extract rating
                    rating = None
                    rating_elem = review_elem.select_one('[class*="rating"], .stars')
                    if rating_elem:
                        try:
                            rating_text = rating_elem.get_text(strip=True)
                            rating_match = re.search(r'[\d.]+', rating_text)
                            if rating_match:
                                rating = float(rating_match.group())
                        except (AttributeError, ValueError):
                            pass
                    
                    # Extract review text
                    text_elem = review_elem.select_one('.review_body, .review_text, [class*="body"]')
                    review_text = text_elem.get_text(strip=True) if text_elem else ""
                    
                    # Extract date
                    date = ""
                    date_elem = review_elem.select_one('time, [class*="date"], .review_date')
                    if date_elem:
                        date = date_elem.get_text(strip=True)
                        # Also try datetime attribute
                        if not date:
                            date = date_elem.get('datetime', '')
                    
                    # Extract helpful count
                    helpful_count = 0
                    helpful_elem = review_elem.select_one('[class*="helpful"], .likes')
                    if helpful_elem:
                        try:
                            helpful_text = helpful_elem.get_text(strip=True)
                            helpful_match = re.search(r'\d+', helpful_text)
                            if helpful_match:
                                helpful_count = int(helpful_match.group())
                        except (AttributeError, ValueError):
                            pass
                    
                    if review_text:  # Only add if there's actual review text
                        reviews.append(Review(
                            username=username,
                            rating=rating,
                            review_text=review_text,
                            date=date,
                            helpful_count=helpful_count
                        ))
                
                # Check for next page
                next_page = soup.select_one('a.next, a[rel="next"], .pagination a:contains("Next")')
                if next_page and len(reviews) < max_reviews:
                    page += 1
                    next_url = next_page.get('href', '')
                    if next_url:
                        if not next_url.startswith('http'):
                            next_url = f"{self.BASE_URL}{next_url}"
                        self.driver.get(next_url)
                        self._wait_for_page_load()
                        self._random_delay()
                else:
                    break
            
            logger.info(f"Scraped {len(reviews)} reviews for {track.title}")
            
        except Exception as e:
            logger.error(f"Error scraping reviews: {e}")
        
        return reviews[:max_reviews]

# ============================================================================
# MAIN SCRAPING FUNCTION
# ============================================================================

def scrape_rym(
    target_tracks: int = 1000,
    max_reviews_per_track: int = 100,
    db_path: str = "rym_music.db",
    headless: bool = False,
    resume: bool = True
):
    """
    Main function to scrape RateYourMusic.
    
    Args:
        target_tracks: Number of tracks to scrape (1k-10k)
        max_reviews_per_track: Maximum reviews to collect per track (up to 100)
        db_path: Path to SQLite database
        headless: Run browser in headless mode
        resume: Resume from last progress if available
    """
    # Initialize database
    conn = create_database(db_path)
    db_manager = DatabaseManager(conn)
    
    # Initialize scraper
    scraper = RYMScraper(db_manager, headless=headless)
    
    # Get progress if resuming
    start_page, last_url, total_scraped = (0, None, 0)
    if resume:
        start_page, last_url, total_scraped = db_manager.get_progress()
        if total_scraped > 0:
            logger.info(f"Resuming from page {start_page}, {total_scraped} tracks already scraped")
    
    tracks_scraped = total_scraped
    current_page = start_page if start_page > 0 else 1
    tracks_per_page = 40  # RYM typically shows 40 items per chart page
    
    try:
        scraper.start()
        skip_until_url = last_url if resume and last_url else None
        
        while tracks_scraped < target_tracks:
            # Fetch chart page
            chart_tracks = scraper.get_chart_page(current_page)
            
            if not chart_tracks:
                logger.warning(f"No tracks found on page {current_page}, trying next page...")
                current_page += 1
                if current_page > 250:  # Safety limit
                    logger.info("Reached maximum page limit")
                    break
                continue
            
            for idx, track_info in enumerate(chart_tracks):
                track_url = track_info['url']
                
                # Skip until we reach the last processed URL (for resuming)
                if skip_until_url:
                    if track_url == skip_until_url:
                        skip_until_url = None
                    continue
                
                # Check if already in database
                if db_manager.track_exists(track_url):
                    logger.info(f"Track already exists: {track_url}")
                    continue
                
                # Calculate chart position
                chart_position = (current_page - 1) * tracks_per_page + idx + 1
                
                # Scrape track details
                track = scraper.scrape_track_page(track_url, chart_position)
                
                if track:
                    # Scrape reviews
                    reviews = scraper.scrape_track_reviews(track, max_reviews_per_track)
                    track.reviews = reviews
                    
                    # Save to database
                    db_manager.insert_track(track)
                    tracks_scraped += 1
                    
                    logger.info(f"[{tracks_scraped}/{target_tracks}] Saved: {track.title}")
                    
                    # Save progress
                    db_manager.save_progress(current_page, track_url, tracks_scraped)
                    
                    # Check if we've reached target
                    if tracks_scraped >= target_tracks:
                        break
                
                # Rate limiting - be respectful to the server
                scraper._random_delay(5, 10)
            
            current_page += 1
            
            # Extra delay between pages
            scraper._random_delay(10, 15)
        
        logger.info(f"Scraping complete! Total tracks: {tracks_scraped}")
        
    except KeyboardInterrupt:
        logger.info("Scraping interrupted by user. Progress has been saved.")
    except Exception as e:
        logger.error(f"Scraping error: {e}")
        raise
    finally:
        scraper.stop()
        conn.close()

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def print_database_stats(db_path: str = "rym_music.db"):
    """Print statistics about the scraped data."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    print("\n" + "=" * 50)
    print("DATABASE STATISTICS")
    print("=" * 50)
    
    tables = ['tracks', 'artists', 'albums', 'genres', 'descriptors', 'reviews']
    
    for table in tables:
        cursor.execute(f"SELECT COUNT(*) FROM {table}")
        count = cursor.fetchone()[0]
        print(f"{table.capitalize():15} {count:,}")
    
    # Additional stats
    cursor.execute("SELECT AVG(average_rating) FROM tracks WHERE average_rating IS NOT NULL")
    avg_rating = cursor.fetchone()[0]
    if avg_rating:
        print(f"\nAverage track rating: {avg_rating:.2f}")
    
    cursor.execute("SELECT MIN(release_year), MAX(release_year) FROM tracks WHERE release_year IS NOT NULL")
    min_year, max_year = cursor.fetchone()
    if min_year and max_year:
        print(f"Year range: {min_year} - {max_year}")
    
    # Top genres
    print("\nTop 10 Genres:")
    cursor.execute("""
        SELECT g.name, COUNT(*) as count 
        FROM genres g 
        JOIN track_genres tg ON g.id = tg.genre_id 
        GROUP BY g.id 
        ORDER BY count DESC 
        LIMIT 10
    """)
    for name, count in cursor.fetchall():
        print(f"  {name}: {count}")
    
    conn.close()
    print("=" * 50 + "\n")


def export_to_json(db_path: str = "rym_music.db", output_path: str = "rym_data.json"):
    """Export database to JSON for analysis."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    data = {"tracks": []}
    
    cursor.execute("SELECT * FROM tracks ORDER BY chart_position")
    tracks = cursor.fetchall()
    
    for track in tracks:
        track_dict = dict(track)
        track_id = track['id']
        
        # Get artists
        cursor.execute("""
            SELECT a.name, a.rym_url, ta.role 
            FROM artists a 
            JOIN track_artists ta ON a.id = ta.artist_id 
            WHERE ta.track_id = ?
        """, (track_id,))
        track_dict['artists'] = [dict(row) for row in cursor.fetchall()]
        
        # Get genres
        cursor.execute("""
            SELECT g.name FROM genres g 
            JOIN track_genres tg ON g.id = tg.genre_id 
            WHERE tg.track_id = ?
        """, (track_id,))
        track_dict['genres'] = [row['name'] for row in cursor.fetchall()]
        
        # Get descriptors
        cursor.execute("""
            SELECT d.name FROM descriptors d 
            JOIN track_descriptors td ON d.id = td.descriptor_id 
            WHERE td.track_id = ?
        """, (track_id,))
        track_dict['descriptors'] = [row['name'] for row in cursor.fetchall()]
        
        # Get reviews
        cursor.execute("""
            SELECT username, rating, review_text, review_date, helpful_count 
            FROM reviews WHERE track_id = ?
        """, (track_id,))
        track_dict['reviews'] = [dict(row) for row in cursor.fetchall()]
        
        data['tracks'].append(track_dict)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)
    
    logger.info(f"Exported data to {output_path}")
    conn.close()


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="RateYourMusic Scraper")
    parser.add_argument("--tracks", type=int, default=1000, help="Number of tracks to scrape (default: 1000)")
    parser.add_argument("--reviews", type=int, default=100, help="Max reviews per track (default: 100)")
    parser.add_argument("--db", type=str, default="rym_music.db", help="Database path")
    parser.add_argument("--no-headless", action="store_true", help="Show browser window")
    parser.add_argument("--no-resume", action="store_true", help="Start fresh, don't resume")
    parser.add_argument("--stats", action="store_true", help="Print database statistics")
    parser.add_argument("--export", type=str, help="Export to JSON file")
    
    args = parser.parse_args()
    
    if args.stats:
        print_database_stats(args.db)
    elif args.export:
        export_to_json(args.db, args.export)
    else:
        scrape_rym(
            target_tracks=args.tracks,
            max_reviews_per_track=args.reviews,
            db_path=args.db,
            headless=not args.no_headless,
            resume=not args.no_resume
        )
