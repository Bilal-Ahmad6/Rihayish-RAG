import argparse
import json
import logging
import re
import math
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple
from urllib.parse import urljoin, urlparse, parse_qs, urlencode, urlunparse

import requests
from bs4 import BeautifulSoup  # type: ignore

# Add parent directory to path to import our local config
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import settings
from scripts.utils import ensure_dirs


# -----------------------
# Logging setup
# -----------------------
LOG_DIR = Path("logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger("graana_scraper")
logger.setLevel(logging.INFO)
_fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

ch = logging.StreamHandler(sys.stdout)
ch.setFormatter(_fmt)
logger.addHandler(ch)

fh = logging.FileHandler(LOG_DIR / "scrape.log", encoding="utf-8")
fh.setFormatter(_fmt)
logger.addHandler(fh)


# -----------------------
# Configurable timings
# -----------------------
POLITE_DELAY_SEC = 1.0
PAGE_DELAY_SEC = 1.5
MAX_RETRIES = 3
TIMEOUT = getattr(settings, "requests_timeout", 20)
USER_AGENT = getattr(settings, "user_agent", "Mozilla/5.0")


# -----------------------
# Optional Selenium fallback
# -----------------------
def try_load_with_selenium(url: str, wait_seconds: float = 3.5) -> Optional[str]:
    try:
        from selenium import webdriver  # type: ignore
        from selenium.webdriver.chrome.options import Options  # type: ignore
        from selenium.webdriver.common.by import By  # type: ignore
        from selenium.webdriver.support.ui import WebDriverWait  # type: ignore
        from selenium.webdriver.support import expected_conditions as EC  # type: ignore
    except Exception:
        logger.warning("Selenium not available. Skipping JS-render fallback for %s", url)
        return None

    try:
        options = Options()
        options.add_argument("--headless=new")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument(f"--user-agent={USER_AGENT}")
        driver = webdriver.Chrome(options=options)
        try:
            driver.set_page_load_timeout(TIMEOUT)
            driver.get(url)
            WebDriverWait(driver, wait_seconds).until(
                EC.presence_of_all_elements_located((By.CSS_SELECTOR, "body"))
            )
            html = driver.page_source
            return html
        finally:
            driver.quit()
    except Exception as e:
        logger.warning("Selenium load failed for %s: %s", url, e)
        return None


# -----------------------
# HTTP helpers
# -----------------------
def http_get_with_retry(url: str) -> Optional[requests.Response]:
    headers = {"User-Agent": USER_AGENT}
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(url, headers=headers, timeout=TIMEOUT)
            if resp.status_code == 200:
                return resp
            logger.warning("GET %s -> %s", url, resp.status_code)
        except Exception as e:
            logger.warning("GET error (attempt %s/%s) %s: %s", attempt, MAX_RETRIES, url, e)
        time.sleep(POLITE_DELAY_SEC * attempt)
    return None


# robots.txt checks intentionally disabled per user request.
def check_robots_allow(urls_to_check: List[str]) -> bool:  # noqa: D401
    """Disabled: always returns True."""
    return True


# -----------------------
# Extraction helpers for Graana
# -----------------------
DETAIL_LINK_PATTERNS = [
    re.compile(r"/property/", re.I),
    re.compile(r"/residential/", re.I),
    re.compile(r"/details/", re.I),
]


def is_property_detail_link(href: Optional[str]) -> bool:
    if not href:
        return False
    href_l = href.lower()
    if href_l.startswith(("javascript:", "mailto:", "#")):
        return False
    # Graana property detail URLs appear as /property/<slug-id>/
    if "/property/" in href_l:
        return True
    return any(p.search(href_l) for p in DETAIL_LINK_PATTERNS)


def collect_detail_links_from_listing(html: str, base_url: str) -> Tuple[List[str], Dict[str, Dict[str, Any]]]:
    """Return list of detail links and a preview info map keyed by property id.

    Preview data includes: price, area, bedrooms, bathrooms, location (best-effort).
    """
    soup = BeautifulSoup(html, "lxml")
    links: List[str] = []
    preview: Dict[str, Dict[str, Any]] = {}

    # Card-level pass: each card tends to have one anchor plus amenity icons
    cards = []
    # Heuristic: containers that include an anchor to /property/ AND at least one img[alt='amenity']
    for candidate in soup.select("div, article, li"):
        if candidate.select_one("a[href*='/property/']") and candidate.select_one("img[alt='amenity']"):
            cards.append(candidate)

    # Fallback to anchors if card detection fails
    anchor_mode = False
    if not cards:
        anchor_mode = True
        cards = soup.select("a[href*='/property/']")  # type: ignore

    for card in cards:
        a = card if anchor_mode else card.select_one("a[href*='/property/']")
        if not a:
            continue
        href = a.get("href")
        href = a.get("href")
        if not is_property_detail_link(href):
            continue
        full = urljoin(base_url, href).split("?")[0]
        links.append(full)
        text = extract_text(card if not anchor_mode else a)
        prop_id_match = re.search(r"-(\d+)/?$", full)
        prop_id = prop_id_match.group(1) if prop_id_match else full
        # Parse price text (keep human format)
        price_match = re.search(r"PKR\s*[0-9,.]+\s*(?:Crore|Lakh)?", text, re.I)
        price = price_match.group(0).strip() if price_match else None
        # Area (Marla/Kanal)
        area_match = re.search(r"(\d+(?:\.\d+)?)\s*(Marla|Kanal)", text, re.I)
        area = area_match.group(0).strip() if area_match else None
        # Bedrooms/Bathrooms/Area via 'amenity N' pattern (Graana uses repeated labels)
        amen_numbers = re.findall(r"amenity\s+(\d+)", text, re.I)
        bedrooms = amen_numbers[0] if len(amen_numbers) >= 1 else None
        bathrooms = amen_numbers[1] if len(amen_numbers) >= 2 else None
        # property type
        type_match = re.search(r"PKR[^A-Za-z]*(House|Flat|Apartment|Plot|Villa|Upper Portion|Lower Portion|Penthouse)", text, re.I)
        property_type = type_match.group(1).title() if type_match else None
        # listing added time
        added_match = re.search(r"Added\s+[^ ]+\s+ago", text, re.I)
        listing_added = added_match.group(0).strip() if added_match else None
        # Location
        loc_match = re.search(r"Bahria Town Phase\s*8[^,]*,?\s*Rawalpindi", text, re.I)
        location = loc_match.group(0).strip() if loc_match else None

        # Enhanced extraction using icon-based amenity groups inside the card container
        container = card if not anchor_mode else a.find_parent(["div", "article", "li"]) if a else None
        if container:
            # Amenity groups: each block containing img[alt=amenity] + numeric sibling(s)
            for img in container.select("img[alt='amenity'], img[src*='bed.svg'], img[src*='bath.svg'], img[src*='area.svg']"):
                src = (img.get("src") or "").lower()
                # Numbers often in sibling div with class MuiTypography-subtitle2New
                num_el = img.find_parent().find_next_sibling("div") if img.find_parent() else None
                if not num_el:
                    # fallback: next text node
                    num_el = img.find_parent().find_next("div") if img.find_parent() else None
                value_txt = extract_text(num_el).strip() if num_el else ""
                value_clean = value_txt.strip().replace("Marla", " Marla").replace("marla", " marla")
                if "bed.svg" in src and value_txt:
                    num_m = re.search(r"(\d+)", value_txt)
                    if num_m:
                        bedrooms = num_m.group(1)
                elif "bath.svg" in src and value_txt:
                    num_m = re.search(r"(\d+)", value_txt)
                    if num_m:
                        bathrooms = num_m.group(1)
                elif "area.svg" in src and value_clean:
                    area_m = re.search(r"(\d+(?:\.\d+)?)\s*([A-Za-z]+)", value_clean)
                    if area_m:
                        num, unit = area_m.groups()
                        area = f"{num} {unit.title()}".strip()
            # Property type lower-case label
            if not property_type:
                pt_candidate = container.select_one("div.MuiTypography-subtitle2New, div[class*='subtitle']")
                if pt_candidate:
                    pt_text = extract_text(pt_candidate).strip().title()
                    if re.fullmatch(r"House|Flat|Apartment|Plot|Villa|Penthouse|Upper Portion|Lower Portion", pt_text, re.I):
                        property_type = pt_text

        preview[prop_id] = {
            "price": price,
            "area": area,
            "bedrooms": bedrooms,
            "bathrooms": bathrooms,
            "location": location,
            "property_type": property_type,
            "listing_added": listing_added,
        }

    # Deduplicate while preserving order
    seen: Set[str] = set()
    ordered: List[str] = []
    for u in links:
        if u not in seen:
            seen.add(u)
            ordered.append(u)
    return ordered, preview


def parse_total_pages(html: str, current_url: str) -> Optional[int]:
    """Infer total number of pages from '( N properties available )' snippet and pageSize param."""
    m = re.search(r"\(\s*([0-9,]+)\s+properties\s+available", html, re.I)
    if not m:
        # fallback pattern without parenthesis
        m = re.search(r"([0-9,]+)\s+properties\s+available", html, re.I)
    if not m:
        return None
    try:
        total_props = int(m.group(1).replace(",", ""))
    except ValueError:
        return None
    # Extract pageSize from URL
    parsed = urlparse(current_url)
    qs = parse_qs(parsed.query)
    page_size = 30
    try:
        if "pageSize" in qs:
            page_size = int(qs["pageSize"][0])
    except Exception:
        pass
    if page_size <= 0:
        return None
    pages = math.ceil(total_props / page_size)
    return pages if pages > 0 else None


def find_next_page_url(current_url: str, html: str) -> Optional[str]:
    soup = BeautifulSoup(html, "lxml")

    # 1) rel="next"
    link = soup.find("a", rel="next")
    if link and link.get("href"):
        return urljoin(current_url, link.get("href"))

    # 2) Pagination list items containing next
    next_link = soup.find("a", string=lambda s: isinstance(s, str) and s.strip().lower() in {"next", ">"})
    if next_link and next_link.get("href"):
        return urljoin(current_url, next_link.get("href"))

    # 3) Graana pagination pattern - increment page parameter
    parsed = urlparse(current_url)
    query_params = parse_qs(parsed.query)
    
    if 'page' in query_params:
        current_page = int(query_params['page'][0])
        next_page = current_page + 1
        query_params['page'] = [str(next_page)]
        new_query = urlencode(query_params, doseq=True)
        return urlunparse(parsed._replace(query=new_query))

    return None


def extract_text(el) -> str:
    return el.get_text(" ", strip=True) if el else ""


def parse_property_details_table(soup: BeautifulSoup) -> Dict[str, str]:
    details: Dict[str, str] = {}
    # Try common table patterns
    for table in soup.select("table, ._1f3d3b1b, ._2b859f59"):
        rows = table.select("tr") or table.select(".row")
        for r in rows:
            cells = r.find_all(["td", "th", "div", "span"], recursive=True)
            if len(cells) >= 2:
                k = extract_text(cells[0])
                v = extract_text(cells[1])
                if k and v:
                    details[k] = v
    return details


def parse_agent_info(soup: BeautifulSoup) -> Dict[str, str]:
    agent_name = "not provided"
    agent_phone = "not provided"

    name_sel = [
        ".agent-name",
        "[class*='agent'] [class*='name']",
        "[class*='Agency'] [class*='name']",
        "[itemprop='name']",
    ]
    for sel in name_sel:
        el = soup.select_one(sel)
        if el and extract_text(el):
            agent_name = extract_text(el)
            break

    # Try tel links first
    tel = soup.select_one("a[href^='tel:']")
    if tel and tel.get("href"):
        agent_phone = tel.get("href").replace("tel:", "").strip() or agent_phone
    else:
        # Try other selectors
        phone_sel = [".phone", "[class*='phone']", "[class*='contact']"]
        for sel in phone_sel:
            el = soup.select_one(sel)
            if el and extract_text(el):
                agent_phone = extract_text(el)
                break

    return {"name": agent_name, "phone": agent_phone}


def parse_amenities(soup: BeautifulSoup) -> List[str]:
    # Look for sections under an Amenities heading
    amenities: List[str] = []
    for heading in soup.find_all(["h2", "h3", "h4"], string=True):
        text = heading.get_text(strip=True).lower()
        if "amenities" in text or "features" in text:
            ul = heading.find_next(["ul", "ol", "div"])
            if ul:
                for li in ul.select("li, .amenity, .feature"):
                    t = extract_text(li)
                    if t:
                        amenities.append(t)
    # Fallback: common classes
    if not amenities:
        for li in soup.select(".amenities li, .features li"):
            t = extract_text(li)
            if t:
                amenities.append(t)
    # Dedup
    seen = set()
    out = []
    for a in amenities:
        if a not in seen:
            seen.add(a)
            out.append(a)
    return out


# Image scraping functions
def extract_images_from_property_page(soup: BeautifulSoup, url: str) -> List[str]:
    """Extract all property images from a Graana property detail page."""
    images: List[str] = []
    
    # Strategy 1: Look for property gallery/carousel images
    # Common selectors for property images on Graana
    image_selectors = [
        "img[src*='property']",
        "img[src*='listing']", 
        "img[src*='graana']",
        ".gallery img",
        ".carousel img",
        ".property-images img",
        ".slider img",
        "[class*='gallery'] img",
        "[class*='carousel'] img",
        "[class*='slider'] img",
        "[class*='image'] img",
        ".swiper-slide img",
        "[data-testid*='image'] img",
        "[data-testid*='gallery'] img"
    ]
    
    seen_images = set()
    
    for selector in image_selectors:
        try:
            elements = soup.select(selector)
            for img in elements:
                src = img.get('src')
                if not src:
                    continue
                
                # Convert relative URLs to absolute
                if src.startswith('//'):
                    src = 'https:' + src
                elif src.startswith('/'):
                    src = urljoin(url, src)
                
                # Filter out non-property images
                if is_property_image(src):
                    if src not in seen_images:
                        seen_images.add(src)
                        images.append(src)
        except Exception as e:
            logger.debug("Error with selector %s: %s", selector, e)
    
    # Strategy 2: Look in script tags for image data (JSON-LD or inline JS)
    script_images = extract_images_from_scripts(soup, url)
    for img_url in script_images:
        if img_url not in seen_images:
            seen_images.add(img_url)
            images.append(img_url)
    
    # Strategy 3: Look for meta property images
    meta_images = extract_images_from_meta(soup, url)
    for img_url in meta_images:
        if img_url not in seen_images:
            seen_images.add(img_url)
            images.append(img_url)
    
    # Remove duplicates while preserving order and filter out small/icon images
    filtered_images = []
    for img_url in images:
        if is_valid_property_image(img_url):
            filtered_images.append(img_url)
    
    return filtered_images


def extract_images_from_scripts(soup: BeautifulSoup, base_url: str) -> List[str]:
    """Extract images from script tags containing JSON data."""
    images = []
    
    # Look for JSON-LD structured data
    for script in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(script.string or "")
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict):
                        images.extend(extract_images_from_json_block(item, base_url))
            elif isinstance(data, dict):
                images.extend(extract_images_from_json_block(data, base_url))
        except Exception as e:
            logger.debug("Error parsing JSON-LD: %s", e)
    
    # Look for inline JavaScript with image arrays
    for script in soup.find_all("script"):
        if script.string:
            script_content = script.string
            # Look for image arrays in JavaScript
            img_patterns = [
                r'"(https?://[^"]*\.(?:jpg|jpeg|png|webp)[^"]*)"',
                r"'(https?://[^']*\.(?:jpg|jpeg|png|webp)[^']*)'",
                r'src:\s*["\']([^"\']*\.(?:jpg|jpeg|png|webp)[^"\']*)["\']',
                r'image:\s*["\']([^"\']*\.(?:jpg|jpeg|png|webp)[^"\']*)["\']',
            ]
            
            for pattern in img_patterns:
                matches = re.findall(pattern, script_content, re.IGNORECASE)
                for match in matches:
                    img_url = match
                    if img_url.startswith('//'):
                        img_url = 'https:' + img_url
                    elif img_url.startswith('/'):
                        img_url = urljoin(base_url, img_url)
                    
                    if is_property_image(img_url):
                        images.append(img_url)
    
    return images


def extract_images_from_json_block(data: Dict[str, Any], base_url: str) -> List[str]:
    """Extract images from a JSON data block."""
    images = []
    
    # Common keys that might contain images
    image_keys = ['image', 'images', 'photo', 'photos', 'picture', 'pictures']
    
    for key in image_keys:
        if key in data:
            value = data[key]
            if isinstance(value, str):
                img_url = value
                if img_url.startswith('//'):
                    img_url = 'https:' + img_url
                elif img_url.startswith('/'):
                    img_url = urljoin(base_url, img_url)
                if is_property_image(img_url):
                    images.append(img_url)
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, str):
                        img_url = item
                        if img_url.startswith('//'):
                            img_url = 'https:' + img_url
                        elif img_url.startswith('/'):
                            img_url = urljoin(base_url, img_url)
                        if is_property_image(img_url):
                            images.append(img_url)
                    elif isinstance(item, dict) and 'url' in item:
                        img_url = item['url']
                        if img_url.startswith('//'):
                            img_url = 'https:' + img_url
                        elif img_url.startswith('/'):
                            img_url = urljoin(base_url, img_url)
                        if is_property_image(img_url):
                            images.append(img_url)
    
    return images


def extract_images_from_meta(soup: BeautifulSoup, base_url: str) -> List[str]:
    """Extract images from meta tags."""
    images = []
    
    # Look for Open Graph and Twitter meta images
    meta_selectors = [
        'meta[property="og:image"]',
        'meta[name="twitter:image"]',
        'meta[property="og:image:url"]',
        'meta[name="twitter:image:src"]'
    ]
    
    for selector in meta_selectors:
        meta = soup.select_one(selector)
        if meta and meta.get('content'):
            img_url = meta['content']
            if img_url.startswith('//'):
                img_url = 'https:' + img_url
            elif img_url.startswith('/'):
                img_url = urljoin(base_url, img_url)
            
            if is_property_image(img_url):
                images.append(img_url)
    
    return images


def is_property_image(url: str) -> bool:
    """Check if URL is likely a property image (not logo, icon, etc.)."""
    if not url:
        return False
    
    url_lower = url.lower()
    
    # Skip common non-property images
    skip_patterns = [
        'logo', 'icon', 'favicon', 'avatar', 'profile',
        'banner', 'header', 'footer', 'nav', 'menu',
        'button', 'arrow', 'close', 'search', 'social',
        'facebook', 'twitter', 'instagram', 'linkedin',
        'whatsapp', 'email', 'phone', 'contact'
    ]
    
    for pattern in skip_patterns:
        if pattern in url_lower:
            return False
    
    # Must be an image file
    if not re.search(r'\.(jpg|jpeg|png|webp)(\?|$)', url_lower):
        return False
    
    # Should contain property-related keywords or be from property domains
    property_indicators = [
        'property', 'listing', 'house', 'home', 'real-estate',
        'graana', 'zameen', 'imarat', 'bahria'
    ]
    
    for indicator in property_indicators:
        if indicator in url_lower:
            return True
    
    # If no clear indicators but looks like a property image URL structure
    if '/property/' in url_lower or '/listing/' in url_lower:
        return True
    
    return False


def is_valid_property_image(url: str) -> bool:
    """Additional validation for property images (size, etc.)."""
    if not is_property_image(url):
        return False
    
    # Skip very small images (likely icons/thumbnails)
    # This is a basic check - could be enhanced with actual image dimension checking
    small_indicators = ['thumb', 'small', 'mini', 'icon', '_s.', '_xs.', '_sm.']
    url_lower = url.lower()
    
    for indicator in small_indicators:
        if indicator in url_lower:
            return False
    
    return True


# Image scraping intentionally disabled per user request.


def derive_parsed_features(description: str) -> Dict[str, bool]:
    text = description.lower()
    features = {
        "near_park": any(kw in text for kw in ["near park", "facing park", "adjacent to park", "park facing"]),
        "corner": "corner" in text,
        "furnished": "furnished" in text,
        "gas": "gas" in text,
        "electricity": "electricity" in text or "wps" in text,
        "water": "water" in text,
        "security": "security" in text or "gated" in text,
        "lawn_garden": any(kw in text for kw in ["lawn", "garden"]),
        "basement": "basement" in text,
        "elevator": any(kw in text for kw in ["elevator", "lift"]),
        "servant_quarter": "servant" in text,
        "boring_bore": any(kw in text for kw in ["boring", "bore"]),
        "parking": any(kw in text for kw in ["parking", "car porch", "garage"]),
    }
    return features


def extract_price_from_page(soup: BeautifulSoup, html: str) -> str:
    """Extract price from Graana detail page using multiple strategies."""
    # Strategy 1: meta tags (og:price:amount not standard but check) or og:title containing PKR
    for meta_name in ["og:title", "twitter:title", "description", "og:description"]:
        el = soup.find("meta", attrs={"property": meta_name}) or soup.find("meta", attrs={"name": meta_name})
        if el and el.get("content") and re.search(r"PKR|Crore|Lakh|Rs", el["content"], re.I):
            m = re.search(r"(PKR[^|]+|Rs[^|]+)", el["content"])
            if m:
                return m.group(1).strip()

    # Strategy 2: common visible selectors
    price_selectors = [
        "[class*='price']",
        "[class*='Price']",
        "[data-testid*='price']",
        "span:contains('PKR')",
    ]
    for sel in price_selectors:
        try:
            el = soup.select_one(sel)
        except Exception:
            el = None
        if el:
            text = extract_text(el)
            if re.search(r"PKR|Crore|Lakh|Rs", text, re.I):
                return text

    # Strategy 3: regex on full HTML (fallback)
    price_pattern = re.compile(r"PKR\s*[0-9,.]+\s*(?:Crore|Lakh)?|[0-9,.]+\s*(?:Crore|Lakh)\b|Rs\.?\s*[0-9,.]+", re.I)
    m = price_pattern.search(html)
    if m:
        return m.group(0).strip()
    return "not provided"


def extract_detail_fields(url: str, use_selenium: bool = False, preview: Optional[Dict[str, Any]] = None, scrape_images: bool = False) -> Dict[str, Any]:
    resp = http_get_with_retry(url)
    html = resp.text if resp else None
    if not html and use_selenium:
        html = try_load_with_selenium(url)
    if not html:
        logger.warning("Failed to load detail page: %s", url)
        html = ""

    soup = BeautifulSoup(html, "lxml")

    # Title: prefer h1 then og:title meta
    title = extract_text(soup.select_one("h1")) or "not provided"
    if title == "not provided":
        og_title = soup.find("meta", property="og:title") or soup.find("meta", attrs={"name": "og:title"})
        if og_title and og_title.get("content"):
            title = og_title["content"].strip()

    price = extract_price_from_page(soup, html)
    if (not price or price == "not provided") and preview and preview.get("price"):
        price = preview["price"]

    # Location: look for pattern 'Phase 8' lines or location class
    location = extract_text(soup.select_one("[class*='location'], [class*='address']"))
    if not location or len(location.split()) < 2:
        # try regex search in text
        loc_match = re.search(r"Bahria Town Phase\s*8[^,]*,?\s*Rawalpindi", html, re.I)
        if loc_match:
            location = loc_match.group(0)
    if not location:
        location = "not provided"
    if location == "not provided" and preview and preview.get("location"):
        location = preview["location"]

    # Normalize text (remove scripts) for regex field extraction
    text_content = " ".join(t for t in soup.stripped_strings if len(t) < 300)

    # JSON-LD structured data (if any)
    json_ld_blocks: List[Dict[str, Any]] = []
    for script in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(script.string or "")
            if isinstance(data, list):
                json_ld_blocks.extend([d for d in data if isinstance(d, dict)])
            elif isinstance(data, dict):
                json_ld_blocks.append(data)
        except Exception:
            continue

    # Extract numeric fields from JSON-LD if present
    for block in json_ld_blocks:
        # Bedrooms / Bathrooms
        if (not preview or not preview.get("bedrooms")) and "numberOfRooms" in block and isinstance(block["numberOfRooms"], (int, str)):
            if str(block["numberOfRooms"]).isdigit():
                preview = preview or {}
                preview["bedrooms"] = str(block["numberOfRooms"])  # type: ignore
        if (not preview or not preview.get("bathrooms")) and block.get("numberOfBathroomsTotal"):
            val = block.get("numberOfBathroomsTotal")
            if isinstance(val, (int, str)) and str(val).isdigit():
                preview = preview or {}
                preview["bathrooms"] = str(val)  # type: ignore
        # Area structured as floorSize / size, etc.
        for key in ["floorSize", "size", "area"]:
            if key in block and isinstance(block[key], dict):
                val = block[key].get("value") or block[key].get("@value")
                unit = block[key].get("unitCode") or block[key].get("unitText")
                if val and unit:
                    area_candidate = f"{val} {unit}"
                    if "Marla" in area_candidate or "Kanal" in area_candidate or unit in {"SQF", "SQFT"}:
                        preview = preview or {}
                        preview["area"] = area_candidate  # type: ignore
        # Price
        if (not preview or not preview.get("price")) and "offers" in block and isinstance(block["offers"], dict):
            off = block["offers"]
            price_val = off.get("price")
            if price_val:
                currency = off.get("priceCurrency", "PKR")
                preview = preview or {}
                preview["price"] = f"{currency} {price_val}"  # type: ignore

    def regex_find(patterns: List[re.Pattern]) -> str:
        for pat in patterns:
            m = pat.search(text_content)
            if m:
                return m.group(1)
        return "not provided"

    bedrooms = regex_find([
        re.compile(r"(\d+)\s*(?:Bed(?:s)?|Bedroom(?:s)?)", re.I),
        re.compile(r'"bedrooms"\s*:\s*(\d+)', re.I),
    ])
    bathrooms = regex_find([
        re.compile(r"(\d+)\s*(?:Bath(?:s)?|Bathroom(?:s)?)", re.I),
        re.compile(r'"bathrooms"\s*:\s*(\d+)', re.I),
    ])
    kitchens = regex_find([
        re.compile(r"(\d+)\s*(?:Kitchen(?:s)?)", re.I),
    ])
    floors = regex_find([
        re.compile(r"(\d+)\s*(?:Floor(?:s)?|Storey|Stories|Story)", re.I),
    ])
    # Fallback to preview for bedroom/bathroom counts
    if (not bedrooms or bedrooms == "not provided") and preview and preview.get("bedrooms"):
        bedrooms = preview["bedrooms"]
    if (not bathrooms or bathrooms == "not provided") and preview and preview.get("bathrooms"):
        bathrooms = preview["bathrooms"]

    # Description: look for a description container else build from paragraphs
    description = "not provided"
    for sel in ["[class*='description']", "[class*='detail']", "article", "section"]:
        el = soup.select_one(sel)
        if el:
            txt = extract_text(el)
            if len(txt.split()) > 12:
                description = txt
                break
    if description == "not provided":
        # fallback combine first few paragraphs
        paras = [extract_text(p) for p in soup.find_all('p')]
        paras = [p for p in paras if len(p.split()) > 3]
        if paras:
            description = " ".join(paras[:4])
    if description == "not provided":
        # meta description
        meta_desc = soup.find("meta", attrs={"name": "description"}) or soup.find("meta", property="og:description")
        if meta_desc and meta_desc.get("content") and len(meta_desc["content"].split()) > 8:
            description = meta_desc["content"].strip()

    # Area extraction
    area_match = re.search(r"(\d+(?:\.\d+)?)\s*(Marla|Kanal|Sq\.?\s*Ft|Square Feet|Sq\.?\s*Yards?)", text_content, re.I)
    area = area_match.group(0) if area_match else "not provided"
    if area == "not provided" and preview and preview.get("area"):
        area = preview["area"]

    # Icon-based extraction on detail page (override if missing or clearly inferior)
    def parse_icon_metrics() -> Dict[str, str]:
        out: Dict[str, str] = {}
        imgs = soup.select("img[alt='amenity'], img[src*='bed.svg'], img[src*='bath.svg'], img[src*='area.svg']")
        for img in imgs:
            src = (img.get("src") or "").lower()
            parent = img.find_parent(["div", "span", "li"]) or img.parent
            value_el = None
            if parent:
                siblings = list(parent.find_all_next(["div", "span"], limit=4))
                for sib in siblings:
                    txt = extract_text(sib).strip()
                    if txt and len(txt) <= 28:
                        value_el = sib
                        break
            value_txt = extract_text(value_el).strip() if value_el else ""
            if not value_txt:
                continue
            if "bed.svg" in src:
                m = re.search(r"(\d+)", value_txt)
                if m:
                    out["bedrooms"] = m.group(1)
            elif "bath.svg" in src:
                m = re.search(r"(\d+)", value_txt)
                if m:
                    out["bathrooms"] = m.group(1)
            elif "area.svg" in src:
                m = re.search(r"(\d+(?:\.\d+)?)\s*([A-Za-z]+)", value_txt)
                if m:
                    out["area"] = f"{m.group(1)} {m.group(2).title()}"
        return out

    icon_metrics = parse_icon_metrics()
    # Override logic: if current field missing or bedrooms == '1' while baths >= '3' assume mis-detected
    try:
        baths_int = int(bathrooms) if bathrooms.isdigit() else None
    except Exception:
        baths_int = None
    if (bedrooms == "not provided" or bedrooms == "1" and baths_int and baths_int >= 3) and icon_metrics.get("bedrooms"):
        bedrooms = icon_metrics["bedrooms"]
    if (bathrooms == "not provided" or not bathrooms.isdigit()) and icon_metrics.get("bathrooms"):
        bathrooms = icon_metrics["bathrooms"]
    if (area == "not provided" or len(area.split()) == 1) and icon_metrics.get("area"):
        area = icon_metrics["area"]

    # Property type & listing added (prefer detail page, else preview)
    property_type = "not provided"
    pt_match = re.search(r"(House|Flat|Apartment|Plot|Villa|Upper Portion|Lower Portion|Penthouse)", text_content, re.I)
    if pt_match:
        property_type = pt_match.group(1).title()
    elif preview and preview.get("property_type"):
        property_type = preview["property_type"]

    listing_added = "not provided"
    la_match = re.search(r"Added\s+[^ ]+\s+ago", text_content, re.I)
    if la_match:
        listing_added = la_match.group(0)
    elif preview and preview.get("listing_added"):
        listing_added = preview["listing_added"]

    # Price normalization
    def normalize_price_to_pkr(price_text: str) -> Optional[int]:
        if not price_text:
            return None
        p = price_text.replace(",", "").strip()
        m = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*(Crore|Lakh)?", p, re.I)
        if not m:
            m2 = re.search(r"PKR\s*([0-9]+)\b", p, re.I)
            if m2:
                try:
                    return int(m2.group(1))
                except Exception:
                    return None
            return None
        num = float(m.group(1))
        unit = m.group(2).lower() if m.group(2) else None
        if unit == "crore":
            return int(num * 10000000)
        if unit == "lakh":
            return int(num * 100000)
        return int(num)

    price_numeric = normalize_price_to_pkr(price)

    # Property ID
    prop_id_match = re.search(r"-(\d+)/?$", url)
    property_id = prop_id_match.group(1) if prop_id_match else "not provided"

    property_details_raw = parse_property_details_table(soup)
    if area == "not provided":
        for k, v in property_details_raw.items():
            if re.search(r"area|size", k, re.I):
                area = v
                break

    amenities = parse_amenities(soup)
    # If amenities empty, attempt icon/feature list fallback
    if not amenities:
        for block in soup.select("[class*='amenity'], [class*='feature'], ul li"):
            txt = extract_text(block)
            if txt and 3 < len(txt) < 60 and txt.lower() not in {a.lower() for a in amenities}:
                # Filter out generic noise
                if not re.search(r"^(pk|rs|[0-9]+)$", txt.strip(), re.I):
                    amenities.append(txt.strip())
    agent = parse_agent_info(soup)
    images: List[str] = extract_images_from_property_page(soup, url) if scrape_images else []
    parsed_features = derive_parsed_features(description if description != "not provided" else "")

    record: Dict[str, Any] = {
        "url": url,
        "title": title or "not provided",
        "price": price or "not provided",
        "area": area or "not provided",
        "location": location or "not provided",
        "bedrooms": bedrooms or "not provided",
        "bathrooms": bathrooms or "not provided",
        "kitchens": kitchens or "not provided",
        "floors": floors or "not provided",
        "description": description or "not provided",
        "parsed_features": parsed_features,
        "amenities": amenities if amenities else [],
        "property_details_raw": property_details_raw,
        "agent": agent,
        "images": images,
    "property_type": property_type,
    "listing_added": listing_added,
    "price_numeric_pkr": price_numeric if price_numeric else None,
    "property_id": property_id,
        "scraped_at": datetime.now(timezone.utc).isoformat(),
    }
    return record


def count_non_empty_fields(rec: Dict[str, Any]) -> int:
    non_empty = 0
    for k, v in rec.items():
        if v is None:
            continue
        if isinstance(v, str) and v.strip().lower() == "not provided":
            continue
        if isinstance(v, (list, dict)) and len(v) == 0:
            continue
        non_empty += 1
    return non_empty


def load_progress(progress_path: Path) -> Dict[str, Any]:
    if progress_path.exists():
        try:
            with progress_path.open("r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            logger.warning("Failed to load progress file, starting fresh: %s", progress_path)
    return {"results": [], "processed_urls": []}


def save_progress(progress_path: Path, results: List[Dict[str, Any]], processed: Iterable[str]) -> None:
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = {"results": results, "processed_urls": list(processed)}
    with progress_path.open("w", encoding="utf-8") as f:
        json.dump(tmp, f, ensure_ascii=False, indent=2)


def scrape_all(start_url: str, max_pages: int, use_selenium: bool, progress_path: Path, refresh_missing: bool=False, force_refresh: bool=False, scrape_images: bool=False) -> List[Dict[str, Any]]:
    logger.info("Start scraping from %s (requested pages=%s)", start_url, max_pages)

    # Robots.txt check
    if not check_robots_allow([start_url]):
        logger.error("robots.txt disallows the start URL. Aborting.")
        return []

    progress = load_progress(progress_path)
    results: List[Dict[str, Any]] = progress.get("results", [])
    processed: Set[str] = set(progress.get("processed_urls", []))

    current_url = start_url
    page_num = 0
    detail_links: List[str] = []

    discovered_total_pages: Optional[int] = None

    preview_accumulator: Dict[str, Dict[str, Any]] = {}

    while current_url and (max_pages <= 0 or page_num < max_pages):
        logger.info("Fetching listing page %s: %s", page_num + 1, current_url)
        resp = http_get_with_retry(current_url)
        if not resp:
            logger.warning("Failed to fetch listing page: %s", current_url)
            break
        html = resp.text
        if discovered_total_pages is None:
            discovered_total_pages = parse_total_pages(html, current_url)
            if discovered_total_pages:
                logger.info("Discovered total pages = %s", discovered_total_pages)
                # If user asked for all pages (<=0), set loop guard
                if max_pages <= 0:
                    max_pages = discovered_total_pages
        if discovered_total_pages and page_num >= discovered_total_pages:
            logger.info("Reached discovered total pages limit (%s)", discovered_total_pages)
            break
        page_links, page_preview = collect_detail_links_from_listing(html, current_url)
        logger.info("Found %s potential detail links on page %s", len(page_links), page_num + 1)
        detail_links.extend(page_links)
        preview_accumulator.update(page_preview)

        page_num += 1
        time.sleep(PAGE_DELAY_SEC)
        next_url = find_next_page_url(current_url, html)
        # If predictable pattern overshoots, double-check with robots (optional)
        current_url = next_url

    # Deduplicate detail links
    seen_links: Set[str] = set()
    ordered_links: List[str] = []
    for u in detail_links:
        if u not in seen_links:
            seen_links.add(u)
            ordered_links.append(u)

    logger.info("Total unique detail links gathered: %s", len(ordered_links))

    # Check robots for a sample property path; if disallowed, warn
    sample_check = ordered_links[:3] if ordered_links else []
    if sample_check and not check_robots_allow(sample_check):
        logger.error("robots.txt disallows property detail pages. Aborting.")
        return results

    # Visit each property detail
    for idx, url in enumerate(ordered_links, start=1):
        if url in processed and not (refresh_missing or force_refresh):
            logger.info("[%s/%s] Skipping already processed: %s", idx, len(ordered_links), url)
            continue
        if url in processed and (refresh_missing or force_refresh):
            # Decide if we should refresh (missing critical fields)
            existing = next((r for r in results if r.get("url") == url), None)
            if existing:
                if force_refresh:
                    logger.info("[%s/%s] Force refreshing: %s", idx, len(ordered_links), url)
                else:
                    critical = ["bedrooms", "bathrooms", "area", "price", "property_type"]
                    all_present = all(existing.get(c) and existing.get(c) != "not provided" for c in critical)
                    # Suspicious heuristic: bedrooms extremely low compared to bathrooms
                    suspicious = False
                    try:
                        b = int(existing.get("bedrooms") or 0)
                        ba = int(existing.get("bathrooms") or 0)
                        if b == 1 and ba >= 3:
                            suspicious = True
                    except Exception:
                        pass
                    if all_present and not suspicious:
                        logger.info("[%s/%s] Already processed & complete: %s", idx, len(ordered_links), url)
                        continue

        logger.info("[%s/%s] Fetching detail: %s", idx, len(ordered_links), url)
        prop_id_match = re.search(r"-(\d+)/?$", url)
        prop_id = prop_id_match.group(1) if prop_id_match else None
        preview = preview_accumulator.get(prop_id) if prop_id else None
        rec = extract_detail_fields(url, use_selenium=use_selenium, preview=preview, scrape_images=scrape_images)
        if url in processed:
            # Replace existing
            for i, r in enumerate(results):
                if r.get("url") == url:
                    results[i] = rec
                    break
        else:
            results.append(rec)
        processed.add(url)
        save_progress(progress_path, results, processed)
        time.sleep(POLITE_DELAY_SEC)

    return results


def write_outputs(results: List[Dict[str, Any]], raw_out_path: Path, template_out_path: Path) -> None:
    raw_out_path.parent.mkdir(parents=True, exist_ok=True)
    with raw_out_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info("Wrote raw JSON array: %s (%s items)", raw_out_path, len(results))

    # Determine max-fields template
    if results:
        best = max(results, key=count_non_empty_fields)
        with template_out_path.open("w", encoding="utf-8") as f:
            json.dump(best, f, ensure_ascii=False, indent=2)
        logger.info("Wrote template with max fields: %s", template_out_path)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Scrape Graana Bahria Town Phase 8 listings and details")
    parser.add_argument("--start-url", default=settings.start_url)
    parser.add_argument("--max-pages", type=int, default=32, help="Number of listing pages to traverse. 0 or negative = ALL (auto-detected)")
    parser.add_argument("--use-selenium", action="store_true", help="Enable Selenium fallback for JS-rendered pages")
    parser.add_argument("--scrape-images", action="store_true", help="Enable image scraping for property listings")
    parser.add_argument("--refresh-missing", action="store_true", help="Re-fetch already processed listings that have missing critical fields (price/bed/bath/area)")
    parser.add_argument("--force-refresh", action="store_true", help="Re-fetch all processed listings regardless of completeness")
    parser.add_argument(
        "--progress", default="data/raw/graana_phase8_progress.json", help="Progress JSON path"
    )
    parser.add_argument(
        "--out", default="data/raw/graana_phase8_raw.json", help="Final raw JSON array output"
    )
    parser.add_argument(
        "--template-out", default="data/raw/template_max_fields.json", help="Template output path"
    )
    args = parser.parse_args(argv)

    ensure_dirs()
    Path("data/raw").mkdir(parents=True, exist_ok=True)

    progress_path = Path(args.progress)
    out_path = Path(args.out)
    template_path = Path(args.template_out)

    results = scrape_all(args.start_url, args.max_pages, args.use_selenium, progress_path, refresh_missing=args.refresh_missing, force_refresh=args.force_refresh, scrape_images=args.scrape_images)
    write_outputs(results, out_path, template_path)

    # Print one sample listing (first element) for schema verification if exists
    if results:
        sample = results[0]
        print("\nSample listing (first element):\n")
        print(json.dumps(sample, ensure_ascii=False, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

