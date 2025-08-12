import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple, Union

# Ensure we can import project modules when running as a script
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import settings  # type: ignore
from utils import ensure_dirs, save_jsonl  # type: ignore
from processor.helpers import normalize_whitespace, parse_float_from_text  # type: ignore


JsonVal = Union[str, int, float, Dict[str, Any], List[Any]]


def parse_price_to_pkr(price_text: str) -> Union[int, str]:
    """Parse a price text into integer PKR.

    Rules:
    - Support formats containing Crore/Lakh/Lac (case-insensitive)
    - Support plain numbers with optional commas and optional Rs/PKR label
    - If cannot parse, return "not provided"
    - KEEP UNIT AS PKR (no conversion to crore/lakh textual units)
    """
    if not price_text or (isinstance(price_text, str) and price_text.strip().lower() == "not provided"):
        return "not provided"

    text = price_text.strip().lower()
    # Remove common labels and extraneous characters, but keep words for crore/lakh detection
    text = text.replace(",", " ")
    text = re.sub(r"pk(?:r)?|rs\.?|rupees|price|approximately|approx\.?", " ", text, flags=re.I)
    text = re.sub(r"\s+", " ", text).strip()

    # Handle Crore / Lakh / Lac variants
    crore_pat = re.compile(r"([0-9]+(?:\.[0-9]+)?)\s*(crore|cr)", re.I)
    lakh_pat = re.compile(r"([0-9]+(?:\.[0-9]+)?)\s*(lakh|lac)", re.I)
    both_pat = re.compile(
        r"(?:(?P<crore>[0-9]+(?:\.[0-9]+)?)\s*(?:crore|cr))?\s*(?:(?P<lakh>[0-9]+(?:\.[0-9]+)?)\s*(?:lakh|lac))?",
        re.I,
    )

    # Case: combined like "1 crore 25 lakh"
    m_both = both_pat.fullmatch(text)
    if m_both and (m_both.group("crore") or m_both.group("lakh")):
        crore_val = float(m_both.group("crore")) if m_both.group("crore") else 0.0
        lakh_val = float(m_both.group("lakh")) if m_both.group("lakh") else 0.0
        pkr = int(round(crore_val * 10_000_000 + lakh_val * 100_000))
        return pkr

    # Single crore/lakh patterns anywhere
    m_crore = crore_pat.search(text)
    if m_crore:
        num = float(m_crore.group(1))
        return int(round(num * 10_000_000))

    m_lakh = lakh_pat.search(text)
    if m_lakh:
        num = float(m_lakh.group(1))
        return int(round(num * 100_000))

    # Grouped numeric like "14,000,000" or "14 000 000"
    m_grouped = re.search(r"\b\d{1,3}(?:[\s,]\d{3})+(?:\.\d+)?\b", text)
    if m_grouped:
        digits_only = re.sub(r"[^0-9]", "", m_grouped.group(0))
        try:
            return int(digits_only)
        except Exception:
            pass

    # Plain numeric (single token)
    m_plain = re.search(r"\b\d+(?:\.\d+)?\b", text)
    if m_plain:
        num_s = m_plain.group(0)
        try:
            if "." in num_s:
                return int(round(float(num_s)))
            return int(num_s)
        except Exception:
            pass

    return "not provided"


def parse_area(area_text: str, property_details: Dict[str, Any]) -> Tuple[str, Union[float, str], Union[float, str]]:
    """Return (unit, value, area_in_marla).

    - unit: one of marla|kanal|sqft|unknown|not provided
    - value: float or "not provided"
    - area_in_marla: float or "not provided" (only kanal converted per requirement)
    """
    if (not area_text) or area_text.strip().lower() == "not provided":
        # Try property_details for area
        for k, v in property_details.items():
            if re.search(r"area", k, re.I):
                area_text = str(v)
                break

    if not area_text:
        return "not provided", "not provided", "not provided"

    text = normalize_whitespace(str(area_text)).lower()
    if not text or text == "not provided":
        return "not provided", "not provided", "not provided"

    # Unit detection
    unit: str = "unknown"
    if "kanal" in text:
        unit = "kanal"
    elif "marla" in text:
        unit = "marla"
    elif re.search(r"sq\s*\.?\s*ft|square\s*feet|sqft", text):
        unit = "sqft"

    value = parse_float_from_text(text)
    if value is None:
        return unit if unit else "unknown", "not provided", "not provided"

    area_in_marla: Union[float, str] = "not provided"
    if unit == "marla":
        area_in_marla = float(value)
    elif unit == "kanal":
        area_in_marla = float(value) * 20.0
    else:
        # sqft or unknown -> no conversion mandated
        area_in_marla = "not provided"

    return unit if unit else "unknown", float(value), area_in_marla


def parse_beds_baths(value_text: str, description: str, kind: str) -> Union[int, str]:
    """Parse integer bedrooms/bathrooms from field text or description."""
    # Try in-value text
    for source in [value_text, description]:
        if not source or source == "not provided":
            continue
        m = re.search(r"(\d{1,2})\s*(%s|%ss?)" % (kind, kind[:-1] if kind.endswith("s") else kind), source, re.I)
        if m:
            try:
                return int(m.group(1))
            except Exception:
                pass
        # fallback: any integer present
        m2 = re.search(r"\b(\d{1,2})\b", source)
        if m2:
            try:
                return int(m2.group(1))
            except Exception:
                pass
    return "not provided"


def dedupe_by_url(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen: set = set()
    unique: List[Dict[str, Any]] = []
    for r in records:
        url = r.get("url") or ""
        if url in seen:
            continue
        seen.add(url)
        unique.append(r)
    return unique


def parse_kitchens(value_text: str, description: str, prop_details: Dict[str, Any]) -> Union[int, str]:
    """Parse integer kitchens from field text, description, or property details."""
    # Try in-value text first
    for source in [value_text, description]:
        if not source or source == "not provided":
            continue
        m = re.search(r"(\d{1,2})\s*(kitchen|kitchens?)", source, re.I)
        if m:
            try:
                return int(m.group(1))
            except Exception:
                pass
    
    # Try property details
    for k, v in prop_details.items():
        if re.search(r"kitchen", k, re.I):
            m = re.search(r"(\d{1,2})", str(v))
            if m:
                try:
                    return int(m.group(1))
                except Exception:
                    pass
    
    return "not provided"


def to_canonical(rec: Dict[str, Any]) -> Dict[str, Any]:
    url = rec.get("url") or ""
    # Derive a stable property/listing id from trailing numeric segment of URL or hash fallback
    prop_id = None
    m = re.search(r"-(\d+)/?$", url)
    if m:
        prop_id = m.group(1)
    else:
        # Hash fallback (short) to avoid collisions if pattern changes
        import hashlib
        prop_id = hashlib.sha1(url.encode("utf-8")).hexdigest()[:12]
    title = rec.get("title") or "not provided"
    price_raw = rec.get("price") or "not provided"
    location = rec.get("location") or "not provided"
    description = rec.get("description") or "not provided"
    images = rec.get("images") or []
    prop_details_raw = rec.get("property_details_raw") or {}

    unit, area_value, area_in_marla = parse_area(rec.get("area") or "not provided", prop_details_raw)

    beds = parse_beds_baths(rec.get("bedrooms") or "", description, "bed")
    kitchens = parse_kitchens(rec.get("kitchens") or "", description, prop_details_raw)

    # Keep only the specified fields
    canonical: Dict[str, Any] = {
        "url": url,
    "property_id": prop_id,
    "listing_id": prop_id,  # keep both names for downstream compatibility
        "title": title,
        "price": price_raw,
        "area": rec.get("area") or "not provided",
        "location": location,
        "bedrooms": beds,
        "kitchens": kitchens,
        "description": description,
        "images": images if isinstance(images, list) else [],
        "price_numeric_pkr": parse_price_to_pkr(price_raw),
    }
    return canonical


def as_text_summary(c: Dict[str, Any], location: str) -> str:
    parts: List[str] = []
    parts.append(f"Title: {c.get('title')}")
    parts.append(f"Price: {c.get('price')}")
    if c.get("area") and c.get("area") != "not provided":
        parts.append(f"Area: {c.get('area')}")
    if location and location != "not provided":
        parts.append(f"Location: {location}")
    parts.append(f"Link: {c.get('url')}")
    return " | ".join(parts)


def as_text_specs(c: Dict[str, Any]) -> str:
    parts: List[str] = []
    parts.append(f"Bedrooms: {c.get('bedrooms')}")
    parts.append(f"Kitchens: {c.get('kitchens')}")
    parts.append(f"Area: {c.get('area')}")
    return " | ".join(parts)


def as_text_features(c: Dict[str, Any]) -> str:
    images = c.get("images")
    images_text = f"Images: {len(images) if isinstance(images, list) else 0}"
    return images_text


def as_text_description(c: Dict[str, Any]) -> str:
    return c.get("description") or "not provided"


def as_text_agent(c: Dict[str, Any]) -> str:
    return f"Price (PKR): {c.get('price_numeric_pkr')} | Link: {c.get('url')}"


def make_chunks(c: Dict[str, Any], base_meta: Dict[str, Any]) -> List[Dict[str, Any]]:
    title = c.get("title")
    url = c.get("url")
    lid = c.get("listing_id") or c.get("property_id")
    price_text = c.get("price")
    location = base_meta.get("location") or "not provided"

    def chunk(chunk_type: str, text: str) -> Dict[str, Any]:
        return {
            "url": url,
            "listing_id": lid,
            "property_id": lid,
            "title": title,
            "price_text": price_text,
            "location": location,
            "chunk_type": chunk_type,
            "text": text,
        }

    return [
        chunk("summary", as_text_summary(c, location)),
        chunk("specs", as_text_specs(c)),
        chunk("features", as_text_features(c)),
        chunk("description", as_text_description(c)),
        chunk("agent", as_text_agent(c)),
    ]


def run(input_path: Path, processed_out: Path, chunks_out: Path) -> Tuple[int, int, int, List[Dict[str, Any]], List[Dict[str, Any]]]:
    ensure_dirs()

    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Handle both direct array format and {"results": [...]} format
    if isinstance(data, dict) and "results" in data:
        raw_records: List[Dict[str, Any]] = data["results"]
    elif isinstance(data, list):
        raw_records: List[Dict[str, Any]] = data
    else:
        raise ValueError("Input file must contain either a JSON array or an object with 'results' key")

    total_raw = len(raw_records)
    unique_raw = dedupe_by_url(raw_records)

    processed: List[Dict[str, Any]] = []
    all_chunks: List[Dict[str, Any]] = []

    for r in unique_raw:
        c = to_canonical(r)
        processed.append(c)
        base_meta = {
            "location": r.get("location") or "not provided",
        }
        all_chunks.extend(make_chunks(c, base_meta))

    # Write outputs
    processed_out.parent.mkdir(parents=True, exist_ok=True)
    with processed_out.open("w", encoding="utf-8") as f:
        json.dump(processed, f, ensure_ascii=False, indent=2)

    save_jsonl(all_chunks, chunks_out)

    return total_raw, len(unique_raw), len(all_chunks), processed, all_chunks


def print_samples(processed: List[Dict[str, Any]], chunks: List[Dict[str, Any]]) -> None:
    if not processed:
        print("No processed listings to sample.")
        return
    first = processed[0]
    sample_listing = {
        "url": first.get("url"),
        "title": first.get("title"),
        "price": first.get("price"),
        "area": first.get("area"),
        "location": first.get("location"),
        "bedrooms": first.get("bedrooms"),
        "kitchens": first.get("kitchens"),
        "description": first.get("description"),
        "images": first.get("images"),
        "price_numeric_pkr": first.get("price_numeric_pkr"),
    }
    print("\nSample processed listing:\n")
    print(json.dumps(sample_listing, ensure_ascii=False, indent=2))

    print("\nFirst 3 chunks:\n")
    for ch in chunks[:3]:
        print(json.dumps(ch, ensure_ascii=False, indent=2))


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Clean, deduplicate, and chunk property listings")
    parser.add_argument("--input", default="data/raw/graana_phase8_with_images.json", type=Path)
    parser.add_argument("--processed-out", default="data/processed/graana_phase8_processed.json", type=Path)
    parser.add_argument("--chunks-out", default="data/processed/graana_phase8_chunks.jsonl", type=Path)
    args = parser.parse_args(argv)

    total_raw, unique_cnt, chunk_cnt, processed, chunks = run(args.input, args.processed_out, args.chunks_out)

    print(
        f"Totals -> raw: {total_raw} | unique(after dedupe): {unique_cnt} | chunks: {chunk_cnt}"
    )
    print_samples(processed, chunks)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

