import argparse
import os

# Auto-load .env so Gemini key (GRAANA_GEMINI_API_KEY) is available when running this script directly
try:  # pragma: no cover - best-effort
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass
import json
import random
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import chromadb
from sentence_transformers import SentenceTransformer  # type: ignore

# Ensure local imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import settings  # type: ignore

# Simple in-process model cache to avoid reload per query
_MODEL_CACHE: Dict[str, SentenceTransformer] = {}

# Query result cache for performance
_QUERY_CACHE: Dict[str, tuple] = {}
_CACHE_MAX_SIZE = 100

# Recently shown properties tracking for better variety
_RECENTLY_SHOWN: List[str] = []  # (Legacy leftover; no longer used after reverting aggressive shuffle)
_RECENTLY_SHOWN_MAX_SIZE = 50

# -----------------------------
# Top-level helper functions (moved out of main so they are reusable
# by answer_one_web and benchmarking utilities)
# -----------------------------

class UserPreferences:
    """Track user preferences across the conversation"""
    def __init__(self):
        self.preferred_locations = []
        self.price_range = {"min": None, "max": None}
        self.property_types = []
        self.size_preferences = []
        self.viewed_properties = []
    
    def update_from_query(self, query: str, analysis: dict):
        """Learn from user interactions"""
        # Track location preferences
        location = analysis.get('filters', {}).get('location')
        if location and location not in self.preferred_locations:
            self.preferred_locations.append(location)
        
        # Track price preferences
        max_price = analysis.get('filters', {}).get('max_price')
        if max_price:
            if not self.price_range["max"] or max_price < self.price_range["max"]:
                self.price_range["max"] = max_price
        
        # Track property type preferences
        prop_type = analysis.get('filters', {}).get('property_type')
        if prop_type and prop_type not in self.property_types:
            self.property_types.append(prop_type)
    
    def get_personalized_filters(self) -> dict:
        """Get filters based on learned preferences"""
        filters = {}
        if self.preferred_locations:
            filters['preferred_locations'] = self.preferred_locations
        if self.price_range["max"]:
            filters['max_price'] = self.price_range["max"]
        if self.property_types:
            filters['property_types'] = self.property_types
        return filters


def get_similar_properties(property_id: str, processed_map: Dict[str, dict], top_n: int = 3) -> List[dict]:
    """Suggest similar properties based on features"""
    if property_id not in processed_map:
        return []
    
    target_property = processed_map[property_id]
    target_price = target_property.get('price_numeric', 0)
    target_bedrooms = target_property.get('bedrooms', '')
    
    similar_properties = []
    
    for lid, prop in processed_map.items():
        if lid == property_id:
            continue
        
        # Calculate similarity score
        score = 0
        prop_price = prop.get('price_numeric', 0)
        
        # Price similarity (within 20% range)
        if target_price > 0 and prop_price > 0:
            price_diff = abs(target_price - prop_price) / target_price
            if price_diff <= 0.2:
                score += 3
            elif price_diff <= 0.5:
                score += 2
        
        # Bedroom similarity
        if target_bedrooms and prop.get('bedrooms') == target_bedrooms:
            score += 2
        
        # Location similarity (same general area)
        target_location = target_property.get('location', '').lower()
        prop_location = prop.get('location', '').lower()
        if target_location and prop_location and target_location in prop_location:
            score += 1
        
        if score > 0:
            similar_properties.append((score, lid, prop))
    
    # Sort by score and return top_n
    similar_properties.sort(key=lambda x: x[0], reverse=True)
    return [{"listing_id": lid, **prop} for _, lid, prop in similar_properties[:top_n]]


def validate_response_quality(query: str, response: str, properties: List[dict]) -> dict:
    """Score response quality and suggest improvements"""
    scores = {}
    
    # Relevance: Do properties match the query intent?
    analysis = enhanced_query_analysis(query)
    filters = analysis.get('filters', {})
    
    relevant_count = 0
    for prop in properties:
        is_relevant = True
        
        # Check price filter
        if 'max_price' in filters:
            prop_price = prop.get('price_numeric', 0)
            if prop_price > filters['max_price'] * 10000000:  # Convert crore to PKR
                is_relevant = False
        
        # Check bedroom filter
        if 'bedrooms' in filters:
            prop_bedrooms = str(prop.get('bedrooms', '')).strip()
            if prop_bedrooms and str(filters['bedrooms']) not in prop_bedrooms:
                is_relevant = False
        
        # Check property type filter (strict matching)
        if 'property_type' in filters:
            prop_title = prop.get('title', '').lower()
            requested_type = filters['property_type'].lower()
            
            if requested_type == 'house':
                # For house requests, exclude apartments/flats
                if any(word in prop_title for word in ['apartment', 'flat', 'unit']):
                    is_relevant = False
                # Must contain house-related terms
                if not any(word in prop_title for word in ['house', 'home', 'villa', 'bungalow', 'marla']):
                    is_relevant = False
            elif requested_type == 'apartment':
                # For apartment requests, exclude houses
                if any(word in prop_title for word in ['house', 'home', 'villa', 'bungalow', 'marla']):
                    is_relevant = False
                # Must contain apartment-related terms
                if not any(word in prop_title for word in ['apartment', 'flat', 'unit']):
                    is_relevant = False
        
        if is_relevant:
            relevant_count += 1
    
    scores['relevance'] = relevant_count / len(properties) if properties else 0
    
    # Completeness: Does response address the query?
    completeness = 0.5  # Base score
    if properties:
        completeness += 0.3
    if len(response.split()) > 10:  # Detailed response
        completeness += 0.2
    scores['completeness'] = min(completeness, 1.0)
    
    # Accuracy: Are property details correct?
    accuracy = 1.0  # Assume accurate unless we detect issues
    for prop in properties:
        if not prop.get('title') or prop.get('title') == 'No title available':
            accuracy -= 0.1
    scores['accuracy'] = max(accuracy, 0.0)
    
    # Overall quality score
    overall_score = sum(scores.values()) / len(scores)
    
    return {
        "scores": scores,
        "overall": overall_score,
        "suggestions": generate_improvement_suggestions(analysis, scores)
    }


def generate_improvement_suggestions(analysis: dict, scores: dict) -> List[str]:
    """Generate suggestions to improve response quality"""
    suggestions = []
    
    if scores.get('relevance', 1) < 0.7:
        suggestions.append("Consider refining search criteria to find more relevant properties")
    
    if scores.get('completeness', 1) < 0.7:
        suggestions.append("Provide more detailed property information")
    
    if analysis['intent'] == 'compare' and len(analysis.get('filters', {})) < 2:
        suggestions.append("Specify comparison criteria (price, location, size, etc.)")
    
    return suggestions


def graceful_error_handling(func):
    """Better error messages and fallback strategies"""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            error_type = type(e).__name__
            if "chroma" in str(e).lower() or "database" in str(e).lower():
                return "I'm having trouble accessing the property database. Please try again in a moment."
            elif "embedding" in str(e).lower() or "model" in str(e).lower():
                return "I couldn't understand your query properly. Could you rephrase it or be more specific?"
            elif "api" in str(e).lower():
                return "I'm experiencing connectivity issues. Please try your request again."
            else:
                return f"I encountered an unexpected issue. Please try rephrasing your question."
    return wrapper


@graceful_error_handling
def enhanced_retrieve(collection, query: str, k: int, embedding_model: str, user_prefs: UserPreferences = None, filters: dict = None) -> Dict[str, List]:
    """Enhanced retrieval with user preferences, filters, and error handling"""
    # Boost query with user preferences
    if user_prefs:
        pref_filters = user_prefs.get_personalized_filters()
        if pref_filters.get('preferred_locations'):
            # Add preferred locations to query for better relevance
            query += f" {' '.join(pref_filters['preferred_locations'])}"
    
    # Enhance query with property type for better retrieval
    if filters and 'property_type' in filters:
        property_type = filters['property_type']
        if property_type == 'house':
            query += " house home villa bungalow marla"
        elif property_type == 'apartment':
            query += " apartment flat unit"
    
    # Use the original retrieve function
    return retrieve(collection, query, k, embedding_model)


def is_casual_greeting_or_irrelevant(query: str) -> bool:
    """Enhanced function to detect greetings and irrelevant queries with better accuracy."""
    if not query or not isinstance(query, str):
        return True
    
    query_lower = query.lower().strip()
    
    # Handle very short queries
    if len(query_lower) <= 1:
        return True

    # Explicit greetings
    greetings = {
        "hi", "hello", "hey", "good morning", "good evening", "good afternoon",
        "how are you", "what's up", "sup", "yo", "greetings", "howdy"
    }

    # Check for exact matches or starting patterns
    if len(query_lower) <= 3 and query_lower in {"hi", "hey", "yo", "sup"}:
        return True
    
    for greeting in greetings:
        if query_lower == greeting or query_lower.startswith(greeting + " "):
            return True

    # Check for property-related keywords - if present, it's NOT irrelevant
    property_keywords = {
        "property", "house", "home", "apartment", "flat", "villa", "bungalow",
        "marla", "kanal", "sqft", "bedroom", "bathroom", "crore", "lakh",
        "price", "cost", "buy", "purchase", "sale", "rent", "listing",
        "bahria", "phase", "block", "real estate", "plot", "land"
    }
    
    # If query contains property keywords, it's relevant
    query_words = set(query_lower.split())
    if any(keyword in query_lower for keyword in property_keywords):
        return False

    # Irrelevant topics
    irrelevant_keywords = {
        "weather", "food", "music", "movie", "game", "sport", "politics",
        "recipe", "joke", "funny", "meme", "cat", "dog", "animal", "news",
        "health", "medicine", "education", "job", "travel", "car", "phone"
    }
    
    # Check if query is clearly about irrelevant topics
    if any(keyword in query_words for keyword in irrelevant_keywords):
        # But allow if it also has property context
        if not any(keyword in query_lower for keyword in property_keywords):
            return True
    
    # Questions without property context
    general_question_starters = ["what is", "who is", "when is", "where is", "why is", "how is"]
    if any(query_lower.startswith(starter) for starter in general_question_starters):
        if not any(keyword in query_lower for keyword in property_keywords):
            return True
    
    return False


def robust_input_validation(query: str) -> tuple[bool, str]:
    """
    Comprehensive input validation with helpful error messages
    Returns (is_valid, error_message)
    """
    # Check if query exists
    if not query:
        return False, "Please provide a question about properties in Bahria Town Phase 8."
    
    # Check if query is string
    if not isinstance(query, str):
        return False, "Invalid input format. Please provide a text query."
    
    # Clean and check length
    cleaned_query = query.strip()
    if len(cleaned_query) < 1:
        return False, "Your query appears to be empty. Please ask about properties you're interested in."
    
    # Check for extremely long queries (potential abuse)
    if len(cleaned_query) > 1000:
        return False, "Your query is too long. Please keep it under 1000 characters."
    
    # Check for obvious spam patterns
    spam_patterns = [
        r'(.)\1{10,}',  # Repeated characters
        r'[^\w\s]{20,}',  # Too many special characters
        r'http[s]?://',  # URLs (basic check)
        r'@\w+',  # Email-like patterns
    ]
    
    import re
    for pattern in spam_patterns:
        if re.search(pattern, cleaned_query, re.IGNORECASE):
            return False, "Please provide a clear property-related question without URLs or spam content."
    
    return True, ""


def get_casual_response(query: str) -> str:
    query_lower = query.lower().strip()
    if any(greet in query_lower for greet in ["hi", "hello", "hey", "good morning", "good evening", "good afternoon"]):
        return (
            "Hello! I'm your real estate assistant for Bahria Town Phase 8. "
            "I can help you find properties based on your preferences.\n\n"
            "Try asking me something like:\n"
            "- Show me 10 marla houses under 5 crores\n"
            "- Find apartments with 2 bedrooms\n"
            "- Find me 10 marla house in Usman Block"
        )
    if "how are you" in query_lower:
        return (
            "I'm doing great, thank you for asking! I'm here and ready to help you find the perfect property in Bahria Town Phase 8.\n\n"
            "What kind of property are you looking for today?"
        )
    return (
        "I'm a specialized real estate assistant for Bahria Town Phase 8 properties. "
        "I can help you search for houses, apartments, and other properties based on your criteria.\n\n"
        "Could you please ask me something related to real estate? For example:\n"
        "- Property size (marla, kanal)\n"
        "- Price range\n"
        "- Number of bedrooms/bathrooms\n"
        "- Location preferences"
    )


def enhanced_query_analysis(query: str) -> dict:
    """Enhanced intent detection and query parsing with robust error handling"""
    if not query or not isinstance(query, str):
        return {"intent": "search", "filters": {}, "original_query": query or ""}
    
    query_lower = query.lower().strip()
    
    # Handle empty or very short queries
    if len(query_lower) < 2:
        return {"intent": "search", "filters": {}, "original_query": query}
    
    # Detect intent with comprehensive patterns
    intent = "search"  # default
    
    # Compare intent (but exclude price range queries)
    compare_patterns = ["compare", "vs", "versus", "difference", "contrast"]
    if any(word in query_lower for word in compare_patterns):
        # Check if it's actually a price range query
        if not any(pattern in query_lower for pattern in ["between", "from", "to", "range"]):
            intent = "compare"
    
    # Analysis intent (but only if NOT a search request)
    analysis_patterns = ["how many total", "total number", "count all", "overall statistics", "market analysis"]
    if any(pattern in query_lower for pattern in analysis_patterns):
        intent = "analyze"
    elif any(word in query_lower for word in ["average price", "mean price", "typical price", "market average"]):
        intent = "analyze"
    
    # Recommendation intent
    recommend_patterns = ["similar to", "like this", "same as", "recommend", "suggest similar"]
    if any(pattern in query_lower for pattern in recommend_patterns):
        intent = "recommend"
    
    # Extract filters with robust error handling
    filters = {}
    
    # Price extraction with comprehensive patterns and error handling
    import re
    
    try:
        # Handle price range queries (from X to Y crores)
        price_range_patterns = [
            r'(?:price\s+range|range)\s+(?:of\s+)?(\d+(?:\.\d+)?)\s*(?:crore|crores?)\s+to\s+(\d+(?:\.\d+)?)\s*(?:crore|crores?)',
            r'(?:between|from)\s+(\d+(?:\.\d+)?)\s*(?:crore|crores?)\s+(?:to|and)\s+(\d+(?:\.\d+)?)\s*(?:crore|crores?)',
            r'(\d+(?:\.\d+)?)\s*(?:crore|crores?)\s+to\s+(\d+(?:\.\d+)?)\s*(?:crore|crores?)',
            r'(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\s*(?:crore|crores?)',
            r'(?:range\s+)?(\d+(?:\.\d+)?)\s*(?:crore|crores?)\s*-\s*(\d+(?:\.\d+)?)\s*(?:crore|crores?)',
            r'between\s+(\d+(?:\.\d+)?)\s+and\s+(\d+(?:\.\d+)?)\s*(?:crore|crores?)',
            r'from\s+(\d+(?:\.\d+)?)\s+to\s+(\d+(?:\.\d+)?)\s*(?:crore|crores?)',
            r'(?:price\s+range|range)\s+(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\s*(?:crore|crores?)'
        ]
        
        price_range_found = False
        for pattern in price_range_patterns:
            matches = re.findall(pattern, query_lower)
            if matches:
                try:
                    min_price = float(matches[0][0])
                    max_price = float(matches[0][1])
                    if 0.1 <= min_price <= max_price <= 100:  # Reasonable range
                        filters['min_price_crores'] = min_price
                        filters['max_price_crores'] = max_price
                        price_range_found = True
                        break
                except (ValueError, IndexError):
                    continue
        
        # If no price range found, check for single price limits
        if not price_range_found:
            # Handle 8-digit numbers as crores (with validation)
            eight_digit_matches = re.findall(r'\b(\d{8})\b', query_lower)
            for match in eight_digit_matches:
                amount = int(match)
                if 10000000 <= amount <= 999999999:  # Valid 8-digit range
                    amount_crores = amount / 10000000
                    filters['max_price_crores'] = amount_crores
                    break
            
            # Handle explicit crore mentions with comprehensive patterns
            crore_patterns = [
                r'under (\d+(?:\.\d+)?)\s*crores?',
                r'below (\d+(?:\.\d+)?)\s*crores?',
                r'less than (\d+(?:\.\d+)?)\s*crores?',
                r'(\d+(?:\.\d+)?)\s*crores?\s*(?:or less|max|maximum|budget|range)',
                r'budget.*?(\d+(?:\.\d+)?)\s*crores?',
                r'(?:my\s+)?budget\s+is\s+(\d+(?:\.\d+)?)\s*crores?',
                r'can\s+afford\s+(\d+(?:\.\d+)?)\s*crores?',
                r'up\s+to\s+(\d+(?:\.\d+)?)\s*crores?',
                r'within\s+(\d+(?:\.\d+)?)\s*crores?',
                r'(\d+(?:\.\d+)?)\s*crores?\s*(?:budget|limit)',
                r'(\d+(?:\.\d+)?)\s*crores?'  # General crore mentions
            ]
            
            for pattern in crore_patterns:
                matches = re.findall(pattern, query_lower)
                if matches:
                    try:
                        amount = float(matches[0])
                        if 0.1 <= amount <= 100:  # Reasonable crore range
                            filters['max_price_crores'] = amount
                            break
                    except (ValueError, IndexError):
                        continue
        
        # Handle lakh mentions with validation
        lakh_patterns = [
            r'under (\d+(?:\.\d+)?)\s*lakhs?',
            r'below (\d+(?:\.\d+)?)\s*lakhs?',
            r'less than (\d+(?:\.\d+)?)\s*lakhs?',
            r'(\d+(?:\.\d+)?)\s*lakhs?\s*(?:or less|max|maximum|budget)',
            r'budget.*?(\d+(?:\.\d+)?)\s*lakhs?',
            r'(?:my\s+)?budget\s+is\s+(\d+(?:\.\d+)?)\s*lakhs?',
            r'(\d+(?:\.\d+)?)\s*lakhs?'
        ]
        
        for pattern in lakh_patterns:
            matches = re.findall(pattern, query_lower)
            if matches:
                try:
                    amount = float(matches[0])
                    if 1 <= amount <= 10000:  # Reasonable lakh range
                        filters['max_price_crores'] = amount / 100  # Convert to crores
                        break
                except (ValueError, IndexError):
                    continue
    
    except Exception:
        # If price extraction fails, continue without price filter
        pass
    
    # Bedroom extraction with validation
    try:
        bedroom_patterns = [
            r'(\d+)\s*(?:bed|bedroom|bedrooms|br|bdr)',
            r'(\d+)\s*(?:room|rooms)',
            r'(\d+)\s*(?:b|bd)'
        ]
        for pattern in bedroom_patterns:
            matches = re.findall(pattern, query_lower)
            if matches:
                try:
                    bedrooms = int(matches[0])
                    if 1 <= bedrooms <= 10:  # Reasonable bedroom range
                        filters['bedrooms'] = bedrooms
                        break
                except (ValueError, IndexError):
                    continue
    except Exception:
        pass
    
    # Area size extraction with validation
    try:
        area_patterns = [
            r'(\d+)\s*marla',
            r'(\d+)\s*sqft',
            r'(\d+)\s*square\s*feet',
            r'(\d+)\s*kanal'
        ]
        for pattern in area_patterns:
            matches = re.findall(pattern, query_lower)
            if matches:
                try:
                    area = int(matches[0])
                    if 1 <= area <= 10000:  # Reasonable area range
                        filters['area_size'] = area
                        break
                except (ValueError, IndexError):
                    continue
    except Exception:
        pass
    
    # Property type extraction with comprehensive patterns
    try:
        house_keywords = ["house", "home", "villa", "bungalow", "mansion", "residence"]
        apartment_keywords = ["apartment", "flat", "unit", "condo", "penthouse"]
        
        property_type = None
        if any(word in query_lower for word in house_keywords):
            property_type = "house"
        elif any(word in query_lower for word in apartment_keywords):
            property_type = "apartment"
        
        if property_type:
            filters['property_type'] = property_type
    except Exception:
        pass
    
    # Extract search terms for title matching with better filtering
    try:
        # Define comprehensive stop words
        stop_words = {
            'show', 'me', 'find', 'get', 'search', 'for', 'in', 'at', 'with', 'under', 'below', 'above', 'over',
            'the', 'a', 'an', 'and', 'or', 'but', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'can', 'may',
            'i', 'you', 'he', 'she', 'it', 'we', 'they', 'this', 'that', 'these', 'those', 'my', 'your',
            'want', 'need', 'looking', 'buy', 'purchase', 'afford', 'budget', 'please', 'help', 'assist'
        }
        
        # Extract meaningful words for title search
        words = re.findall(r'\b[a-zA-Z]+\b', query_lower)
        search_terms = [word for word in words if word not in stop_words and len(word) > 2]
        
        # Remove price/bedroom/area related words from search terms
        filter_words = {
            'crore', 'crores', 'lakh', 'lakhs', 'million', 'price', 'cost', 'under', 'below', 'above',
            'bed', 'bedroom', 'bedrooms', 'bd', 'bdr', 'room', 'rooms',
            'marla', 'sqft', 'square', 'feet', 'kanal', 'area', 'size'
        }
        
        search_terms = [word for word in search_terms if word not in filter_words]
        
        # Only keep search terms if they're meaningful
        if search_terms:
            filters['title_search_terms'] = search_terms
    except Exception:
        filters['title_search_terms'] = []
    
    # Extract requested number of properties with validation
    try:
        # Handle written numbers and articles
        word_to_number = {
            'a': 1, 'an': 1, 'one': 1, 'single': 1,
            'two': 2, 'three': 3, 'four': 4, 'five': 5,
            'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10,
            'eleven': 11, 'twelve': 12, 'fifteen': 15, 'twenty': 20
        }
        
        # Look for written numbers with property keywords
        for word, num in word_to_number.items():
            patterns = [
                rf'\b{word}\s+(?:house|apartment|property|listing|properties|homes|units|place|option)\b',
                rf'\b{word}\s+(?:bedroom|bed|br)\b',  # "one bedroom"
                rf'\bshow\s+me\s+{word}\b',  # "show me one"
                rf'\bfind\s+me\s+{word}\b',  # "find me one"
                rf'\bfind\s+{word}\b',  # "find one"
                rf'\bgive\s+me\s+{word}\b',  # "give me one"
                rf'\bwant\s+{word}\b',  # "want two"
                rf'\bneed\s+{word}\b',  # "need one"
                rf'\blooking\s+for\s+{word}\b',  # "looking for one"
                rf'\b{word}\s+(?:good|nice|suitable)\b'  # "one good house"
            ]
            
            for pattern in patterns:
                matches = re.findall(pattern, query_lower)
                if matches:
                    try:
                        filters['requested_count'] = num
                        break
                    except:
                        continue
            if 'requested_count' in filters:
                break
        
        # Look for numeric patterns if no written number found
        if 'requested_count' not in filters:
            number_patterns = [
                r'show\s+me\s+(\d+)',
                r'find\s+me\s+(\d+)',
                r'find\s+(\d+)',
                r'get\s+(\d+)',
                r'give\s+me\s+(\d+)',
                r'(\d+)\s+(?:properties|listings|houses|apartments|flats|homes|options)',
                r'top\s+(\d+)',
                r'first\s+(\d+)',
                r'(\d+)\s+(?:house|apartment|property)',
                r'need\s+(\d+)',
                r'want\s+(\d+)'
            ]
            
            for pattern in number_patterns:
                matches = re.findall(pattern, query_lower)
                if matches:
                    try:
                        count = int(matches[0])
                        if 1 <= count <= 50:  # Reasonable count range
                            filters['requested_count'] = count
                            break
                    except (ValueError, IndexError):
                        continue
    except Exception:
        pass
    
    return {
        "intent": intent,
        "filters": filters,
        "original_query": query
    }


def title_based_search(processed_map: Dict[str, dict], search_terms: List[str], filters: Dict[str, Any]) -> List[dict]:
    """
    Enhanced title-based search with robust filtering and error handling
    Returns properties sorted by relevance score and filters applied
    """
    if not processed_map or not isinstance(processed_map, dict):
        return []
    
    scored_properties = []
    
    try:
        for listing_id, property_data in processed_map.items():
            if not property_data or not isinstance(property_data, dict):
                continue
                
            title = str(property_data.get('title', '')).lower()
            
            # Calculate title relevance score
            title_score = 0
            if search_terms and isinstance(search_terms, list):
                for term in search_terms:
                    if term and isinstance(term, str):
                        # Exact word matches get higher score
                        if f" {term.lower()} " in f" {title} ":
                            title_score += 2
                        # Partial matches get lower score
                        elif term.lower() in title:
                            title_score += 1
            
            # Apply filters with robust error handling
            passes_filters = True
            
            # Price filter with validation (both min and max)
            if 'max_price_crores' in filters or 'min_price_crores' in filters:
                try:
                    property_price_pkr = property_data.get('price_numeric_pkr', 0)
                    
                    # Handle different price field formats
                    if not property_price_pkr:
                        property_price_pkr = property_data.get('price_numeric', 0)
                    
                    if property_price_pkr and isinstance(property_price_pkr, (int, float)) and property_price_pkr > 0:
                        property_price_crores = property_price_pkr / 10000000  # Convert PKR to crores
                        
                        # Check maximum price
                        if 'max_price_crores' in filters:
                            max_price_crores = float(filters['max_price_crores'])
                            if property_price_crores > max_price_crores:
                                passes_filters = False
                        
                        # Check minimum price
                        if 'min_price_crores' in filters:
                            min_price_crores = float(filters['min_price_crores'])
                            if property_price_crores < min_price_crores:
                                passes_filters = False
                    else:
                        # If no valid price data, exclude from price-filtered searches only if price was specifically requested
                        if filters.get('max_price_crores', 100) < 100 or filters.get('min_price_crores', 0) > 0:
                            passes_filters = False
                except (ValueError, TypeError, ZeroDivisionError):
                    # If price processing fails, exclude only if price was specifically requested
                    passes_filters = False
            
            # Bedroom filter with validation
            if 'bedrooms' in filters:
                try:
                    requested_bedrooms = int(filters['bedrooms'])
                    property_bedrooms = property_data.get('bedrooms')
                    
                    # Handle different bedroom field formats
                    if isinstance(property_bedrooms, str):
                        if property_bedrooms.isdigit():
                            property_bedrooms = int(property_bedrooms)
                        elif property_bedrooms.lower() in ['not provided', 'n/a', '']:
                            property_bedrooms = None
                        else:
                            # Try to extract number from string
                            import re
                            bedroom_match = re.search(r'\d+', property_bedrooms)
                            if bedroom_match:
                                property_bedrooms = int(bedroom_match.group())
                            else:
                                property_bedrooms = None
                    
                    if property_bedrooms != requested_bedrooms:
                        passes_filters = False
                except (ValueError, TypeError):
                    passes_filters = False
            
            # Property type filter with enhanced matching
            if 'property_type' in filters:
                try:
                    requested_type = str(filters['property_type']).lower()
                    title_lower = title.lower()
                    
                    if requested_type == 'house':
                        house_keywords = ['house', 'home', 'villa', 'bungalow', 'mansion', 'residence', 'marla']
                        if not any(word in title_lower for word in house_keywords):
                            passes_filters = False
                    elif requested_type == 'apartment':
                        apartment_keywords = ['apartment', 'flat', 'unit', 'condo', 'penthouse']
                        if not any(word in title_lower for word in apartment_keywords):
                            passes_filters = False
                except (TypeError, AttributeError):
                    passes_filters = False
            
            # Area filter with enhanced parsing
            if 'area_size' in filters:
                try:
                    requested_area = int(filters['area_size'])
                    area_str = str(property_data.get('area', ''))
                    
                    # Try multiple area fields
                    if not area_str or area_str.lower() in ['not provided', 'n/a']:
                        area_str = str(property_data.get('area_unit', ''))
                    
                    if area_str:
                        import re
                        area_matches = re.findall(r'\d+', area_str)
                        if area_matches:
                            property_area = int(area_matches[0])
                            # Allow some tolerance (±1 for marla, ±100 for sqft)
                            if 'marla' in area_str.lower():
                                if abs(property_area - requested_area) > 1:
                                    passes_filters = False
                            elif 'sqft' in area_str.lower() or 'square' in area_str.lower():
                                if abs(property_area - requested_area) > 100:
                                    passes_filters = False
                            else:
                                if property_area != requested_area:
                                    passes_filters = False
                        else:
                            passes_filters = False
                    else:
                        passes_filters = False
                except (ValueError, TypeError, IndexError):
                    passes_filters = False
            
            # Only include properties that pass all filters
            if passes_filters:
                # If no search terms provided or search terms are empty, include all filtered properties
                # If search terms provided, require title match OR if no title matches found, include anyway
                include_property = False
                
                if not search_terms or len(search_terms) == 0:
                    # No search terms, include all filtered properties
                    include_property = True
                elif title_score > 0:
                    # Title matches search terms
                    include_property = True
                else:
                    # Title doesn't match, but we might still want to include it
                    # if the filters are very specific (like price + property type)
                    filter_count = len([k for k in filters.keys() if k != 'title_search_terms'])
                    if filter_count >= 2:  # Multiple specific filters
                        include_property = True
                
                if include_property:
                    try:
                        price_for_sorting = property_data.get('price_numeric_pkr', 0)
                        if not price_for_sorting:
                            price_for_sorting = property_data.get('price_numeric', 0)
                        
                        scored_properties.append({
                            'property': property_data,
                            'score': title_score,
                            'listing_id': listing_id,
                            'price_for_sorting': price_for_sorting or 0
                        })
                    except Exception:
                        # If there's an error adding the property, skip it but continue
                        continue
        
        # Sort by title relevance score (highest first), then by price (lowest first)
        try:
            scored_properties.sort(key=lambda x: (-x['score'], x['price_for_sorting']))
        except Exception:
            # If sorting fails, return unsorted results
            pass
        
        # Keep ordering by score then price for relevance, but introduce mild variety:
        # shuffle only within groups of identical score so relative relevance tiers preserved.
        try:
            i = 0
            while i < len(scored_properties):
                j = i + 1
                base_score = scored_properties[i]['score'] if isinstance(scored_properties[i], dict) else None
                while j < len(scored_properties) and isinstance(scored_properties[j], dict) and scored_properties[j]['score'] == base_score:
                    j += 1
                # Now range [i:j) shares the same score
                if j - i > 1:
                    random.shuffle(scored_properties[i:j])
                i = j
        except Exception:
            pass  # On any issue, fallback silently to sorted order
        
        return [item['property'] for item in scored_properties]
    
    except Exception as e:
        # If entire function fails, return empty list but log the error
        print(f"Error in title_based_search: {str(e)}")
        return []


def format_property_listing(property_data: dict) -> str:
    """Format a single property listing with all required fields"""
    if not property_data:
        return ""
    
    # Extract and clean title
    title = property_data.get('title', 'No title available')
    if '|' in title:
        title = title.split('|')[0].strip()
    
    # Price
    price = property_data.get('price', 'Price not available')
    
    # Bedrooms 
    bedrooms = property_data.get('bedrooms')
    bedrooms_str = f"{bedrooms} bedrooms" if bedrooms and bedrooms != "not provided" else ""
    
    # Kitchens
    kitchens = property_data.get('kitchens')
    kitchens_str = f"{kitchens} kitchens" if kitchens and kitchens != "not provided" else ""
    
    # Description
    description = property_data.get('description', '')
    if description and description != "not provided":
        # Limit description length
        if len(description) > 200:
            description = description[:200] + "..."
    else:
        description = ""
    
    # URL
    url = property_data.get('url', '#')
    
    # Images
    images = property_data.get('images', [])
    has_images = len(images) > 0
    
    # Format the listing
    listing_parts = []
    listing_parts.append(f"🏠 **{title}**")
    listing_parts.append(f"💰 **Price:** {price}")
    
    # Only add fields that have valid data
    if bedrooms_str:
        listing_parts.append(f"🛏️ **Bedrooms:** {bedrooms_str}")
    
    if kitchens_str:
        listing_parts.append(f"🍳 **Kitchens:** {kitchens_str}")
    
    if description:
        listing_parts.append(f"📝 **Description:** {description}")
    
    listing_parts.append(f"🔗 **View Property:** [Visit Listing]({url})")
    
    if has_images:
        listing_parts.append(f"📸 **Images:** {len(images)} photos available [View Images Button]")
    
    return "\n".join(listing_parts)


def generate_search_response(query: str, properties: List[dict], filters: Dict[str, Any]) -> str:
    """Generate a conversational response for search results without detailed listings"""
    
    total_found = len(properties)
    requested_count = filters.get('requested_count')
    
    if total_found == 0:
        return "I couldn't find any properties matching your criteria. Please try adjusting your search terms or filters."
    
    # Determine how many to show
    if requested_count:
        show_count = min(requested_count, total_found)
    else:
        # Default to showing 5 properties
        show_count = min(5, total_found)
    
    # Generate simple response message without detailed property listings
    if requested_count and show_count == requested_count:
        if total_found == show_count:
            response = f"I found {total_found} properties matching your criteria. Here are the {show_count} properties you requested:"
        else:
            response = f"I found {total_found} properties matching your criteria. Here are the {show_count} properties you requested:"
    elif total_found > show_count:
        response = f"I found {total_found} properties matching your criteria. Here are the top {show_count} results:"
    else:
        response = f"I found {total_found} properties matching your criteria:"
    
    # Add helpful context if filters were applied
    filter_info = []
    
    # Handle price range or single price limit
    if 'min_price_crores' in filters and 'max_price_crores' in filters:
        min_price = filters['min_price_crores']
        max_price = filters['max_price_crores']
        filter_info.append(f"between {min_price} and {max_price} crores")
    elif 'max_price_crores' in filters:
        filter_info.append(f"under {filters['max_price_crores']} crores")
    elif 'min_price_crores' in filters:
        filter_info.append(f"above {filters['min_price_crores']} crores")
    
    if 'property_type' in filters:
        filter_info.append(f"{filters['property_type']}s")
    if 'bedrooms' in filters:
        filter_info.append(f"{filters['bedrooms']} bedroom")
    if 'area_size' in filters:
        filter_info.append(f"{filters['area_size']} marla")
    
    if filter_info:
        response += f" All properties match your criteria: {', '.join(filter_info)}."
    
    return response


def handle_special_query_types(query: str, processed_map: Dict[str, dict]) -> tuple[bool, dict]:
    """
    Handle special query types that don't fit standard search patterns
    Returns (handled, response_dict)
    """
    query_lower = query.lower().strip()
    
    # Help queries
    help_patterns = ["help", "how to", "what can you do", "commands", "options"]
    if any(pattern in query_lower for pattern in help_patterns):
        response = {
            "answer": """I'm Rihayish, your real estate assistant for Bahria Town Phase 8. I can help you with:

🏠 **Property Search**
• Find houses, apartments, or other properties
• Filter by price range (e.g., "under 5 crores")
• Filter by bedrooms (e.g., "3 bedroom houses")
• Filter by area size (e.g., "10 marla properties")

💡 **Example Queries**
• "Show me 5 houses under 3 crores"
• "Find 2 bedroom apartments"
• "Properties in Usman Block"
• "10 marla houses with 4 bedrooms"

📊 **Market Analysis**
• Ask for property statistics and market insights
• Compare different property types
• Get price ranges and averages

Just ask me what you're looking for and I'll help you find the perfect property!""",
            "mode": "help",
            "listings": []
        }
        return True, response
    
    # Comparison queries without specific properties
    comparison_patterns = ["compare", "difference between", "vs", "versus"]
    if any(pattern in query_lower for pattern in comparison_patterns):
        if not any(prop in query_lower for prop in ["house", "apartment", "property", "marla", "bedroom"]):
            response = {
                "answer": "I can help you compare properties! Please specify what you'd like to compare, such as:\n• 'Compare 2 bedroom vs 3 bedroom houses'\n• 'Difference between apartments and houses'\n• 'Compare 5 marla vs 10 marla properties'",
                "mode": "comparison_help",
                "listings": []
            }
            return True, response
    
    # Location-only queries
    location_patterns = ["where is", "location of", "address of"]
    if any(pattern in query_lower for pattern in location_patterns):
        response = {
            "answer": "All properties in my database are located in Bahria Town Phase 8, Rawalpindi. If you're looking for specific properties, please ask me to search for them with your criteria like price range, bedrooms, or property type.",
            "mode": "location_info",
            "listings": []
        }
        return True, response
    
    # Contact/business queries
    contact_patterns = ["contact", "phone", "email", "office", "address", "business hours"]
    if any(pattern in query_lower for pattern in contact_patterns):
        response = {
            "answer": "I'm an AI assistant that helps you search for properties in Bahria Town Phase 8. For specific property inquiries or to schedule viewings, please use the contact information provided with each property listing. I can help you find properties that match your criteria!",
            "mode": "contact_info",
            "listings": []
        }
        return True, response
    
    # Price-only queries without search intent (but exclude price range queries)
    if "price" in query_lower and not any(word in query_lower for word in ["find", "show", "search", "under", "below", "above", "budget", "afford", "between", "from", "to", "range"]):
        response = {
            "answer": "I can help you find properties within your price range! Please tell me:\n• Your budget (e.g., 'under 5 crores')\n• Property type (house, apartment)\n• Any other preferences (bedrooms, area size)\n\nFor example: 'Show me houses under 3 crores with 3 bedrooms'",
            "mode": "price_help",
            "listings": []
        }
        return True, response
    
    return False, {}


def get_query_suggestions(query: str, filters: dict) -> List[str]:
    """Generate helpful query suggestions based on failed search"""
    suggestions = []
    
    # Budget-based suggestions
    if 'max_price_crores' in filters:
        price = filters['max_price_crores']
        suggestions.append(f"Try increasing your budget above {price} crores")
        suggestions.append(f"Search for properties under {price + 1} crores")
    
    # Bedroom-based suggestions
    if 'bedrooms' in filters:
        bedrooms = filters['bedrooms']
        if bedrooms > 1:
            suggestions.append(f"Try searching for {bedrooms - 1} bedroom properties")
        suggestions.append(f"Try searching for {bedrooms + 1} bedroom properties")
    
    # Property type suggestions
    if 'property_type' in filters:
        current_type = filters['property_type']
        other_type = "apartments" if current_type == "house" else "houses"
        suggestions.append(f"Try searching for {other_type} instead")
    
    # Area suggestions
    if 'area_size' in filters:
        area = filters['area_size']
        suggestions.append(f"Try searching for different plot sizes")
        if area == 10:
            suggestions.append("Try searching for 5 marla or 1 kanal properties")
    
    # General suggestions
    suggestions.extend([
        "Remove some filters to see more results",
        "Try searching without specific location requirements",
        "Ask me to show you all available properties"
    ])
    
    return suggestions[:3]  # Return top 3 suggestions


def handle_irrelevant_query(query: str) -> str:
    """Handle queries that are not related to property search"""
    query_lower = query.lower().strip()
    
    # Greetings
    if any(greet in query_lower for greet in ["hi", "hello", "hey", "good morning", "good evening", "good afternoon"]):
        return (
            "Hello! I'm Rihayish, your real estate assistant for Bahria Town Phase 8. "
            "I can help you find properties based on your specific requirements.\n\n"
            "Try asking me something like:\n"
            "• Show me 3 houses in Usman Block\n"
            "• Find apartments under 2 crores\n"
            "• Show me 5 bedroom houses\n"
            "• Find 10 marla properties under 5 crores"
        )
    
    # How are you
    if "how are you" in query_lower:
        return (
            "I'm doing great, thank you for asking! I'm here and ready to help you find the perfect property in Bahria Town Phase 8.\n\n"
            "What type of property are you looking for today?"
        )
    
    # General questions
    general_questions = ["what", "why", "how", "when", "where", "who"]
    if any(word in query_lower.split() for word in general_questions) and not any(prop_word in query_lower for prop_word in ["house", "property", "apartment", "flat", "marla", "crore", "bedroom"]):
        try:
            # Try to provide a helpful response for general questions
            return f"I understand you're asking: '{query}'. While I'm specifically designed to help you find properties in Bahria Town Phase 8, I'll do my best to help.\n\nHowever, my core purpose is to assist you with property searches. I can help you find houses, apartments, and other properties based on your budget, size preferences, and location within Bahria Town Phase 8.\n\nWould you like to search for any properties?"
        except:
            return "I'm PropertyGuru, specialized in helping you find properties in Bahria Town Phase 8. Please ask me about houses, apartments, or other real estate options you're looking for!"
    
    return "I'm Rihayish, your dedicated real estate assistant for Bahria Town Phase 8. I specialize in helping you find the perfect property based on your requirements like location, price, size, and number of bedrooms.\n\nPlease ask me about properties you'd like to find!"
    """Handle queries that are not related to property search"""
    query_lower = query.lower().strip()
    
    # Greetings
    if any(greet in query_lower for greet in ["hi", "hello", "hey", "good morning", "good evening", "good afternoon"]):
        return (
            "Hello! I'm Rihayish, your real estate assistant for Bahria Town Phase 8. "
            "I can help you find properties based on your specific requirements.\n\n"
            "Try asking me something like:\n"
            "• Show me 3 houses in Usman Block\n"
            "• Find apartments under 2 crores\n"
            "• Show me 5 bedroom houses\n"
            "• Find 10 marla properties under 5 crores"
        )
    
    # How are you
    if "how are you" in query_lower:
        return (
            "I'm doing great, thank you for asking! I'm here and ready to help you find the perfect property in Bahria Town Phase 8.\n\n"
            "What type of property are you looking for today?"
        )
    
    # General questions
    general_questions = ["what", "why", "how", "when", "where", "who"]
    if any(word in query_lower.split() for word in general_questions) and not any(prop_word in query_lower for prop_word in ["house", "property", "apartment", "flat", "marla", "crore", "bedroom"]):
        try:
            # Try to provide a helpful response for general questions
            return f"I understand you're asking: '{query}'. While I'm specifically designed to help you find properties in Bahria Town Phase 8, I'll do my best to help.\n\nHowever, my core purpose is to assist you with property searches. I can help you find houses, apartments, and other properties based on your budget, size preferences, and location within Bahria Town Phase 8.\n\nWould you like to search for any properties?"
        except:
            return "I'm PropertyGuru, specialized in helping you find properties in Bahria Town Phase 8. Please ask me about houses, apartments, or other real estate options you're looking for!"
    
    return "I'm Rihayish, your dedicated real estate assistant for Bahria Town Phase 8. I specialize in helping you find the perfect property based on your requirements like location, price, size, and number of bedrooms.\n\nPlease ask me about properties you'd like to find!"


def extract_requested_number(query: str) -> Optional[int]:
    """Extract the number of properties requested from the query."""
    query_lower = query.lower().strip()
    
    # Check for written numbers first
    word_to_number = {
        'one': 1, 'a': 1, 'single': 1, 'an': 1,
        'two': 2, 'three': 3, 'four': 4, 'five': 5,
        'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10
    }
    
    # Look for written numbers with property keywords
    import re
    for word, num in word_to_number.items():
        patterns = [
            rf'\b{word}\s+(?:house|apartment|property|listing|properties|homes|units|place|option)\b',
            rf'\b{word}\s+(?:bedroom|bed|br)\b',  # "one bedroom"
            rf'\bshow\s+me\s+{word}\b',  # "show me one"
            rf'\bfind\s+{word}\b',  # "find one"
            rf'\bgive\s+me\s+{word}\b',  # "give me one"
            rf'\bwant\s+{word}\b',  # "want two"
            rf'\bneed\s+{word}\b',  # "need one"
        ]
        for pattern in patterns:
            if re.search(pattern, query_lower):
                return num
    
    # Check for digit numbers
    number_match = re.search(r'\b(\d+)\s*(?:house|apartment|property|listing|properties|homes|units)', query_lower)
    if number_match:
        return int(number_match.group(1))
    
    # Special cases for single property requests
    single_indicators = [
        r'\ba\s+(?:house|apartment|property|home|place)',
        r'\ban\s+(?:apartment|option)',
        r'\bshow\s+me\s+(?:a|an)\b',
        r'\bfind\s+(?:a|an)\b',
        r'\bany\s+(?:house|apartment|property)\b'
    ]
    
    for pattern in single_indicators:
        if re.search(pattern, query_lower):
            return 1
    
    return None


def should_use_context(current_query: str, history: List[dict]) -> bool:
    """Intelligent decision on when to use conversation history"""
    if not history:
        return False
    
    query_lower = current_query.lower().strip()
    
    # Use context for pronouns and references
    pronouns = ["it", "that", "this", "them", "those", "these"]
    if any(pronoun in query_lower.split() for pronoun in pronouns):
        return True
    
    # Use context for comparative queries
    comparatives = ["cheaper", "bigger", "smaller", "better", "similar", "like that", "more"]
    if any(comp in query_lower for comp in comparatives):
        return True
    
    # Use context for follow-up location queries (very short)
    location_words = ["in", "at", "near", "around", "from"]
    if len(query_lower.split()) <= 2 and any(word in query_lower for word in location_words):
        return True
    
    # Use context for continuation words
    continuation_words = ["more", "another", "also", "additionally", "plus"]
    if any(word in query_lower for word in continuation_words):
        return True
    
    return False


def normalize_location_query(query: str) -> str:
    """Normalize location terms to improve matching."""
    query_lower = query.lower()
    
    # Location normalization mappings
    location_mappings = {
       
        'phase 8': 'bahria town phase 8',
        'phase8': 'bahria town phase 8',
        'p8': 'phase 8',
        
        'bahria phase 8': 'bahria town phase 8',
       
    }
    
    normalized = query_lower
    for variant, canonical in location_mappings.items():
        if variant in normalized:
            normalized = normalized.replace(variant, canonical)
    
    # Also create an expanded query with location variations for better embedding match
    location_expansions = []
    if 'river hill' in normalized:
        location_expansions.extend(['river hills', 'riverhills', 'river hill'])
    if 'spring north' in normalized:
        location_expansions.extend(['spring north', 'springnorth', 'bahria spring north'])
    if 'spring south' in normalized:
        location_expansions.extend(['spring south', 'springsouth', 'bahria spring south'])
    
    if location_expansions:
        # Add variations to help with embedding matching
        normalized = f"{normalized} {' '.join(location_expansions)}"
    
    return normalized


def is_statistical_query(query: str) -> bool:
    """
    Determine if a query is asking for statistical analysis rather than property search.
    Statistical queries ask for aggregated data, comparisons, or market analysis.
    Search queries look for specific properties, even with budget constraints.
    """
    query_lower = query.lower()
    
    # Search indicators that override statistical classification
    search_indicators = [
        "show me", "find me", "suggest", "recommend", "looking for",
        "want to buy", "need", "search for", "budget is", "can afford",
        "in my budget", "within budget", "my price range"
    ]
    
    # If it's clearly a search request, it's not statistical
    if any(indicator in query_lower for indicator in search_indicators):
        return False
    
    # True statistical analysis keywords
    statistical_keywords = [
        "average price", "mean price", "typical price", "usual price", 
        "market average", "price trend", "market analysis",
        "total count", "how many properties", "number of properties",
        "compare all", "overall market", "market statistics",
        "price distribution", "market report"
    ]
    
    return any(keyword in query_lower for keyword in statistical_keywords)


def get_all_property_data_for_analysis(processed_map: Dict[str, dict], query: str) -> List[dict]:
    query_lower = query.lower()
    all_properties: List[dict] = []
    for listing_id, data in processed_map.items():
        property_info = {
            "listing_id": listing_id,
            "title": data.get("title", ""),
            "price_numeric": data.get("price_numeric", 0),
            "price_raw": data.get("price_raw", ""),
            "bedrooms": data.get("bedrooms", ""),
            "bathrooms": data.get("bathrooms", ""),
            "area_unit": data.get("area_unit", ""),
            "area_size": data.get("area_size", ""),
            "url": data.get("url", ""),
            "scraped_at": data.get("processed_at", ""),
        }
        include_property = True
        if "10 marla" in query_lower and "marla" in str(data.get("area_unit", "")).lower():
            include_property = True
        elif "marla" in query_lower and "marla" not in str(data.get("area_unit", "")).lower():
            include_property = False
        if "house" in query_lower and "apartment" in str(data.get("title", "")).lower():
            include_property = False
        elif "apartment" in query_lower and "house" in str(data.get("title", "")).lower():
            include_property = False
        if include_property and property_info["price_numeric"] and property_info["price_numeric"] > 0:
            all_properties.append(property_info)
    return all_properties


def has_relevant_property_results(res: Dict[str, List], threshold: float = 1.2) -> bool:
    distances = res.get("distances", [[]])[0]
    if not distances:
        return False
    best_distance = min(distances) if distances else 1.0
    return best_distance < threshold

# Web API function for Flask integration
def answer_one_web(query: str, gemini_api_key: str, gemini_model: str = "gemini-1.5-flash") -> str:
    """Simplified web-compatible answer function (string only) using Gemini only."""
    result = rag_infer(query=query, gemini_api_key=gemini_api_key, gemini_model=gemini_model)
    return result.get("answer", "")


def rag_infer(
    query: str,
    gemini_api_key: str = None,
    gemini_model: str = "gemini-1.5-flash",
    gemini_analytical_model: str = "gemini-1.5-pro",
    embedding_model: str = "sentence-transformers/all-mpnet-base-v2",
    k: int = 5,
    conversation_history: List[dict] = None,
) -> Dict[str, object]:
    """Enhanced RAG inference system with comprehensive error handling and robustness.

    Returns a dict containing: answer, listings, mode, etc.
    """
    if conversation_history is None:
        conversation_history = []
        
    out: Dict[str, object] = {"query": query}
    
    try:
        # Comprehensive input validation
        is_valid, error_message = robust_input_validation(query)
        if not is_valid:
            out["answer"] = error_message
            out["mode"] = "input_error"
            out["listings"] = []
            return out
        
        # Clean the query
        query = query.strip()
        
        # Validate parameters
        if k <= 0:
            k = 5
        if k > 50:
            k = 50
            
        # Load processed data with error handling
        script_dir = Path(__file__).parent.parent
        processed_path = script_dir / "data" / "processed" / "graana_phase8_processed.json"
        
        try:
            processed_map = load_processed_map(processed_path)
        except Exception as e:
            out["answer"] = "I'm having trouble accessing the property database. Please try again in a moment."
            out["mode"] = "database_error"
            out["listings"] = []
            return out
        
        out["processed_loaded"] = bool(processed_map)
        if not processed_map:
            out["answer"] = "The property database is currently unavailable. Please try again later."
            out["mode"] = "database_empty"
            out["listings"] = []
            return out

        # Enhanced query analysis with error handling
        try:
            analysis = enhanced_query_analysis(query)
            filters = analysis.get('filters', {})
            intent = analysis.get('intent', 'search')
        except Exception as e:
            # Fallback to basic search if analysis fails
            analysis = {"intent": "search", "filters": {}, "original_query": query}
            filters = {}
            intent = "search"
        
        # Handle special query types first
        handled, special_response = handle_special_query_types(query, processed_map)
        if handled:
            return special_response
        
        # Handle irrelevant queries
        try:
            if is_casual_greeting_or_irrelevant(query):
                out["answer"] = handle_irrelevant_query(query)
                out["mode"] = "greeting"
                out["listings"] = []
                return out
        except Exception:
            # If greeting detection fails, continue with normal processing
            pass
        
        # Handle statistical queries
        try:
            if is_statistical_query(query):
                all_properties = get_all_property_data_for_analysis(processed_map, query)
                if not all_properties:
                    out["answer"] = "I don't have enough property data to perform this analysis. Please try a property search instead."
                    out["mode"] = "stats_no_data"
                    out["listings"] = []
                    return out
                
                # Generate statistical analysis
                analysis_prompt = f"""You are analyzing real estate data for Bahria Town Phase 8. Here is the complete property dataset:\n\n{json.dumps(all_properties[:50], indent=2)}\n\nUser Question: {query}\n\nProvide a detailed analysis with:\n1. Direct answer to their question (with calculated numbers)\n2. Key statistics (average, range, etc.)\n3. Notable insights\n4. Be specific and use the actual data provided."""
                
                if gemini_api_key:
                    try:
                        stats_answer = call_llm_gemini(analysis_prompt, gemini_analytical_model, gemini_api_key, 500)
                        out["answer"] = stats_answer
                    except Exception as e:
                        out["answer"] = f"I have the property data but couldn't generate the analysis. Found {len(all_properties)} properties to analyze."
                else:
                    out["answer"] = "Statistical analysis requires Gemini API key."
                
                out["mode"] = "stats"
                out["listings"] = []
                return out
        except Exception:
            # If statistical processing fails, continue with search
            pass

        # Perform title-based search with error handling
        try:
            search_terms = filters.get('title_search_terms', [])
            properties = title_based_search(processed_map, search_terms, filters)
        except Exception as e:
            properties = []
        
        # Fallback to embedding search if title search fails or returns no results
        if not properties and search_terms:
            try:
                # Initialize ChromaDB client
                chroma_client = chromadb.PersistentClient(path=str(script_dir / "chromadb_data"))
                
                # Get embedding for the query
                embedding = embed_query(embedding_model, query)
                
                # Get collection
                from config import settings as _settings
                collection = chroma_client.get_collection(_settings.collection_name)
                res = collection.query(query_embeddings=[embedding], n_results=k)
                
                if has_relevant_property_results(res):
                    # Process embedding results and apply filters
                    docs = res.get("documents", [[]])[0]
                    metas = res.get("metadatas", [[]])[0]
                    
                    fallback_properties = []
                    seen_ids = set()
                    
                    for meta in metas:
                        if isinstance(meta, dict) and meta.get("listing_id"):
                            listing_id = meta["listing_id"]
                            if listing_id in processed_map and listing_id not in seen_ids:
                                seen_ids.add(listing_id)
                                prop = processed_map[listing_id]
                                
                                # Apply same filters as title search
                                passes_filters = True
                                
                                # Price filter (both min and max)
                                if 'max_price_crores' in filters or 'min_price_crores' in filters:
                                    try:
                                        property_price_pkr = prop.get('price_numeric_pkr', 0) or prop.get('price_numeric', 0)
                                        if property_price_pkr:
                                            property_price_crores = property_price_pkr / 10000000
                                            
                                            # Check maximum price
                                            if 'max_price_crores' in filters:
                                                max_price_crores = float(filters['max_price_crores'])
                                                if property_price_crores > max_price_crores:
                                                    passes_filters = False
                                            
                                            # Check minimum price
                                            if 'min_price_crores' in filters:
                                                min_price_crores = float(filters['min_price_crores'])
                                                if property_price_crores < min_price_crores:
                                                    passes_filters = False
                                        else:
                                            passes_filters = False
                                    except:
                                        passes_filters = False
                                
                                # Bedroom filter
                                if 'bedrooms' in filters:
                                    try:
                                        requested_bedrooms = int(filters['bedrooms'])
                                        property_bedrooms = prop.get('bedrooms')
                                        if isinstance(property_bedrooms, str) and property_bedrooms.isdigit():
                                            property_bedrooms = int(property_bedrooms)
                                        if property_bedrooms != requested_bedrooms:
                                            passes_filters = False
                                    except:
                                        passes_filters = False
                                
                                # Property type filter
                                if 'property_type' in filters:
                                    try:
                                        requested_type = filters['property_type']
                                        title_lower = prop.get('title', '').lower()
                                        
                                        if requested_type == 'house':
                                            if not any(word in title_lower for word in ['house', 'home', 'villa', 'bungalow', 'marla']):
                                                passes_filters = False
                                        elif requested_type == 'apartment':
                                            if not any(word in title_lower for word in ['apartment', 'flat', 'unit']):
                                                passes_filters = False
                                    except:
                                        passes_filters = False
                                
                                if passes_filters:
                                    fallback_properties.append(prop)
                    
                    properties = fallback_properties
                    
            except Exception as e:
                # If embedding search fails, continue with empty results
                pass
        
        # Handle no results with intelligent suggestions
        if not properties:
            suggestions = get_query_suggestions(query, filters)
            suggestion_text = "\n• ".join(suggestions) if suggestions else "adjust your search criteria"
            
            out["answer"] = f"I couldn't find any properties matching your criteria.\n\nYou might want to try:\n• {suggestion_text}"
            out["mode"] = "no_results"
            out["listings"] = []
            out["suggestions"] = suggestions
            return out
        
        # Mild+ shuffling: keep very top result fixed, shuffle remainder of top-score group,
        # then lightly shuffle remaining tail while preserving overall relevance tiers.
        try:
            if not filters.get('requested_count') and len(properties) > 3:
                # Derive scores if available via helper (title_based_search produced ordered list by score).
                # We don't have direct scores here, so approximate top-score group by identical title patterns / price closeness.
                # Simpler heuristic: treat first 3 items that share same cleaned title tokens (ignoring numbers) as top group.
                def norm_title(t: str) -> str:
                    t = (t or '').lower()
                    import re
                    t = re.sub(r'\b(\d+|marla|house|for sale|rawalpindi|bahria|phase|sector|town|graana\.com|\|)\b','', t)
                    t = re.sub(r'[^a-z]+',' ', t).strip()
                    return t
                base_norm = norm_title(properties[0].get('title',''))
                top_group_end = 1
                for idx in range(1, min(len(properties), 6)):
                    if norm_title(properties[idx].get('title','')) == base_norm:
                        top_group_end = idx + 1
                    else:
                        break
                fixed = properties[0:1]
                top_group_remainder = properties[1:top_group_end]
                tail = properties[top_group_end:]
                if len(top_group_remainder) > 1:
                    random.shuffle(top_group_remainder)
                # Light shuffle tail: only if reasonably large
                if len(tail) > 2:
                    # Shuffle a copy of tail but keep relative order for first two in tail for stability
                    tail_head = tail[:2]
                    tail_rest = tail[2:]
                    random.shuffle(tail_rest)
                    tail = tail_head + tail_rest
                properties = fixed + top_group_remainder + tail
        except Exception:
            pass  # Fallback silently
        
        # Apply requested count limit with validation
        try:
            requested_count = filters.get('requested_count')
            if requested_count and isinstance(requested_count, int) and requested_count > 0:
                properties = properties[:min(requested_count, len(properties))]
            else:
                # Default to showing 5 properties
                properties = properties[:min(5, len(properties))]
        except Exception:
            properties = properties[:5]
        
        # Generate conversational response
        try:
            response_text = generate_search_response(query, properties, filters)
        except Exception as e:
            response_text = f"I found {len(properties)} properties matching your criteria."
        
        # Final shuffle to ensure even the limited set is randomized
        random.shuffle(properties)
        
        # Format properties for frontend with error handling
        formatted_listings = []
        for prop in properties:
            try:
                formatted_listing = {
                    'title': str(prop.get('title', 'No title available'))[:200],  # Limit title length
                    'price': str(prop.get('price', 'Price not available'))[:100],
                    'bedrooms': prop.get('bedrooms') if prop.get('bedrooms') not in ["not provided", None, ""] else None,
                    'kitchens': prop.get('kitchens') if prop.get('kitchens') not in ["not provided", None, ""] else None,
                    'description': str(prop.get('description', ''))[:500] if prop.get('description') not in ["not provided", None, ""] else '',
                    'url': str(prop.get('url', '#')),
                    'images': prop.get('images', []) if isinstance(prop.get('images'), list) else [],
                    'property_id': str(prop.get('property_id', '')),
                    'area': str(prop.get('area', '')),
                    'location': str(prop.get('location', ''))
                }
                formatted_listings.append(formatted_listing)
            except Exception:
                # If formatting a property fails, skip it but continue with others
                continue
        
        out["answer"] = response_text
        out["listings"] = formatted_listings
        out["mode"] = "search_results"
        out["total_found"] = len(properties)
        out["filters_applied"] = filters
        
        return out
        
    except Exception as e:
        # Final catch-all error handler
        error_msg = "I encountered an unexpected issue while processing your request. Please try again or rephrase your question."
        out["answer"] = error_msg
        out["mode"] = "system_error"
        out["listings"] = []
        return out


def choose_device() -> str:
    try:
        import torch  # type: ignore

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def load_processed_map(processed_path: Path) -> Dict[str, dict]:
    if not processed_path.exists():
        return {}
    try:
        with processed_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return {}
    out: Dict[str, dict] = {}
    for rec in data:
        lid = rec.get("listing_id")
        if lid:
            out[str(lid)] = rec
    return out


def embed_query(model_name: str, text: str) -> List[float]:
    if model_name in _MODEL_CACHE:
        model = _MODEL_CACHE[model_name]
    else:
        device = choose_device()
        model = SentenceTransformer(model_name, device=device)
        _MODEL_CACHE[model_name] = model
    vec = model.encode([text], batch_size=1, convert_to_numpy=False)[0]
    return vec.tolist() if hasattr(vec, "tolist") else list(vec)


def retrieve(
    collection_name: str, question: str, k: int, embedding_model: str
) -> Dict[str, List]:
    """Retrieve with caching for better performance"""
    # Check cache first
    cache_key = f"{question}_{k}_{embedding_model}"
    if cache_key in _QUERY_CACHE:
        return _QUERY_CACHE[cache_key]
    
    client = chromadb.PersistentClient(path=str(settings.chroma_persist_dir))
    try:
        # Prefer getting existing collection to avoid accidental creation with mismatched schemas
        if hasattr(client, "get_collection"):
            collection = client.get_collection(name=collection_name)
        else:
            collection = client.get_or_create_collection(name=collection_name)
        q_emb = embed_query(embedding_model, question)
        res = collection.query(
            query_embeddings=[q_emb], n_results=k, include=["metadatas", "documents", "distances"]
        )
        
        # Cache the result
        if len(_QUERY_CACHE) >= _CACHE_MAX_SIZE:
            # Remove oldest entry
            oldest_key = next(iter(_QUERY_CACHE))
            del _QUERY_CACHE[oldest_key]
        
        _QUERY_CACHE[cache_key] = res
        return res
    except Exception as e:
        print(
            "Warning: retrieval failed (" + str(e) + ").\n"
            "If this is a Chroma schema error, try deleting the 'chromadb_data' folder and re-running embeddings."
        )
        # Return empty result structure
        return {"documents": [[]], "metadatas": [[]], "distances": [[]]}


def build_context(res: Dict[str, List], processed_map: Dict[str, dict], top_n: int = 5) -> Tuple[str, List[dict]]:
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0]

    lines: List[str] = []
    seen_listings: set = set()
    picked: List[dict] = []

    for i, (doc, meta, dist) in enumerate(zip(docs, metas, dists)):
        lid = (meta or {}).get("listing_id") if isinstance(meta, dict) else None
        if not lid:
            continue
        # Add chunk line
        chunk_type = (meta or {}).get("chunk_type")
        price = (meta or {}).get("price_numeric")
        br = (meta or {}).get("bedrooms")
        ba = (meta or {}).get("bathrooms")
        au = (meta or {}).get("area_unit")
        url = (meta or {}).get("url")
        title = (processed_map.get(lid, {}) or {}).get("title")
        price_raw = (processed_map.get(lid, {}) or {}).get("price_raw")
        lines.append(
            f"[dist={dist:.4f}] [{chunk_type}] listing_id={lid} | title={title} | price={price_raw or price} | br={br} ba={ba} area_unit={au} | url={url} | text={ (doc or '')[:240] }"
        )

        # Track top unique listings with comprehensive details
        if lid not in seen_listings and len(picked) < top_n:
            seen_listings.add(lid)
            
            # Get comprehensive property details
            processed_listing = processed_map.get(lid, {})
            
            picked.append({
                "listing_id": lid,
                "title": title or "No title available",
                "price": price_raw or price or "Price not available",
                "location": processed_listing.get("location", "Bahria Town Phase 7"),
                "url": url or "URL not available",
                "scraped_at": ((processed_listing.get("raw", {}) or {}).get("scraped_at")
                or processed_listing.get("processed_at", "Unknown")),
            })

    # Don't shuffle when user requests specific number - maintain relevance order
    # random.shuffle(picked)  # Removed to respect user's specific requests
    
    context = "\n".join(lines[: top_n * 2])  # include up to 2 chunks per listing
    return context, picked


def call_llm_llamacpp(prompt: str, model_path: str, max_tokens: int = 384) -> str:
    try:
        from llama_cpp import Llama  # type: ignore
    except Exception:
        return "[LLM unavailable: please install llama-cpp-python and provide a GGUF model path]"
    llm = Llama(model_path=model_path, n_ctx=4096, n_threads=6)
    out = llm(prompt=prompt, max_tokens=max_tokens, temperature=0.2)
    return out.get("choices", [{}])[0].get("text", "").strip()


def call_llm_gpt4all(prompt: str, model_name: str, max_tokens: int = 384) -> str:
    try:
        from gpt4all import GPT4All  # type: ignore
    except Exception:
        return "[LLM unavailable: please install gpt4all and provide a local model name]"
    gpt = GPT4All(model_name)
    with gpt.chat_session():
        return gpt.generate(prompt, max_tokens=max_tokens, temp=0.2)


def call_llm_ollama(prompt: str, model_name: str, max_tokens: int = 384) -> str:
    try:
        import requests
    except Exception:
        return "[LLM unavailable: please install requests library]"
    
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model_name,
                "prompt": prompt,
                "options": {
                    "num_predict": max_tokens,
                    "temperature": 0.2
                },
                "stream": False
            },
            timeout=60
        )
        response.raise_for_status()
        result = response.json()
        return result.get("response", "").strip()
    except Exception as e:
        return f"[Ollama unavailable: {str(e)}]"


## Legacy Groq/Cohere helpers removed; Gemini is the sole remote LLM path.


def call_llm_gemini(prompt: str, model_name: str, api_key: str, max_tokens: int = 384) -> str:
    """Call Google Gemini models using the google-generativeai SDK.

    We keep temperature low for deterministic, instruction-following behavior.
    """
    try:
        import google.generativeai as genai  # type: ignore
    except Exception:
        return "[LLM unavailable: please install google-generativeai]"

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name)
        # Use a small safety/decoding config suitable for short RAG summaries
        resp = model.generate_content(
            prompt,
            generation_config={
                "temperature": 0.2,
                "top_p": 0.9,
                "top_k": 40,
                "max_output_tokens": max_tokens,
            },
            safety_settings={
                # Rely on defaults; can be customized later.
            },
        )
        # Handle candidates / safety blocks
        if not resp or not getattr(resp, "text", None):
            return "[Gemini response empty]"
        return resp.text.strip()
    except Exception as e:
        return f"[Gemini unavailable: {str(e)}]"


def call_llm(prompt: str, engine: str, model_name: str, api_key: str, max_tokens: int = 384) -> str:
    """Unified LLM calling function that supports multiple engines."""
    return call_llm_gemini(prompt, model_name, api_key, max_tokens)


def format_answer(question: str, listings: List[dict], freshness_note: str, answer_text: str) -> str:
    """Format the answer with clean property details."""
    
    if not listings:
        return f"\n{answer_text}\n\nNo matching properties found.\n\n{freshness_note}"
    
    # Build detailed property listings with clean formatting
    lines = [f"\n{answer_text}\n"]
    lines.append("MATCHING PROPERTIES:")
    lines.append("=" * 60)
    
    for i, listing in enumerate(listings, 1):
        title = listing.get('title', 'No title available')
        price = listing.get('price', 'Price not available')
        url = listing.get('url', 'URL not available')
        scraped_at = listing.get('scraped_at', 'Unknown')
        
        lines.append(f"\nProperty #{i}")
        lines.append(f"Title: {title}")
        lines.append(f"Price: {price}")
        
        # Property specifications
        lines.append("Specifications:")
        bedrooms = listing.get('bedrooms', 'Not specified')
        bathrooms = listing.get('bathrooms', 'Not specified') 
        area_unit = listing.get('area_unit', 'Not specified')
        area_size = listing.get('area_size', 'Not specified')
        property_type = listing.get('property_type', 'Not specified')
        location = listing.get('location', 'Bahria Town Phase 7')
        
        lines.append(f"  Bedrooms: {bedrooms}")
        lines.append(f"  Bathrooms: {bathrooms}")
        lines.append(f"  Area: {area_size} {area_unit}")
        lines.append(f"  Type: {property_type}")
        lines.append(f"  Location: {location}")
        
        lines.append(f"Link: {url}")
        lines.append(f"Data Date: {scraped_at}")
        
        if i < len(listings):
            lines.append("-" * 60)
    
    lines.append(f"\n{freshness_note}")
    
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="RAG query using Chroma + local LLM")
    parser.add_argument("--collection", default=settings.collection_name)
    parser.add_argument("--query")
    parser.add_argument("--k", type=int, default=5, help="Top-N chunks to retrieve")
    parser.add_argument(
        "--embedding-model",
        default="sentence-transformers/all-mpnet-base-v2",
        help="SentenceTransformers model for query embedding",
    )
    # Default processed dataset updated to Phase 8 Graana file
    parser.add_argument("--processed", type=Path, default=Path("data/processed/graana_phase8_processed.json"))
    parser.add_argument("--llm-engine", choices=["llama", "gpt4all", "ollama", "gemini", "none"], default="gemini")
    parser.add_argument("--llama-model-path", help="Path to GGUF model for llama.cpp", default=None)
    parser.add_argument("--gpt4all-model", help="Local GPT4All model name", default="orca-mini-3b.gguf2.Q4_0.gguf")
    parser.add_argument("--ollama-model", help="Ollama model name", default="llama3.1:8b-instruct-q4_K_M")
    parser.add_argument("--gemini-api-key", help="Gemini API key", default=None)
    parser.add_argument("--gemini-model", help="Gemini model name", default="gemini-1.5-flash")
    parser.add_argument("--gemini-analytical-model", help="Gemini model for analytical/compare queries", default="gemini-1.5-pro")
    parser.add_argument("--explain", action="store_true", help="Print retrieved chunks and scores")
    parser.add_argument("--interactive", action="store_true", help="Interactive chat loop")
    args = parser.parse_args()

    # Fallback: if Gemini key not passed explicitly, pull from env (supports both legacy & new names)
    if not args.gemini_api_key:
        args.gemini_api_key = (
            os.getenv("GRAANA_GEMINI_API_KEY")
            or os.getenv("GEMINI_API_KEY")
            or os.getenv("GOOGLE_API_KEY")  # google-generativeai also checks this; expose for clarity
        )
        if not args.gemini_api_key:
            print("[warn] Gemini API key not provided. Set GRAANA_GEMINI_API_KEY or pass --gemini-api-key to enable LLM answers.")

    processed_map = load_processed_map(args.processed)

    # (helpers now imported from top-level definitions above)
    
    def call_llm_for_general_chat(prompt: str) -> str:
        """Call LLM for general conversation without property context."""
        if args.llm_engine == "llama" and args.llama_model_path:
            return call_llm_llamacpp(prompt, args.llama_model_path)
        elif args.llm_engine == "gpt4all":
            return call_llm_gpt4all(prompt, args.gpt4all_model)
        elif args.llm_engine == "ollama":
            return call_llm_ollama(prompt, args.ollama_model)
        elif args.llm_engine == "gemini" and args.gemini_api_key:
            return call_llm_gemini(prompt, args.gemini_model, args.gemini_api_key)
        else:
            return "I'm a real estate assistant. Enable an LLM engine (gemini, ollama, gpt4all, or llama)."

    def answer_one(q: str, conversation_history: List[dict] = None) -> str:
        if conversation_history is None:
            conversation_history = []
        
        # Initialize user preferences (in a real app, this would be persistent)
        user_prefs = UserPreferences()
        
        # Enhanced query analysis
        analysis = enhanced_query_analysis(q)
        user_prefs.update_from_query(q, analysis)
            
        # Handle obvious greetings first
        if is_casual_greeting_or_irrelevant(q):
            casual_response = get_casual_response(q)
            return casual_response
        
        # Check if this is a statistical/analytical query
        if is_statistical_query(q):
            # Get all property data for analysis
            all_properties = get_all_property_data_for_analysis(processed_map, q)
            
            if all_properties:
                # Create context for statistical analysis
                stats_context = f"Property Data Analysis for query: '{q}'\n\n"
                stats_context += "Available Properties:\n"
                
                for i, prop in enumerate(all_properties[:50], 1):  # Limit to 50 for context
                    stats_context += f"{i}. {prop['title']} - PKR {prop['price_numeric']:,} - {prop['area_size']} {prop['area_unit']}\n"
                
                stats_context += f"\nTotal properties found: {len(all_properties)}"
                
                # Use LLM to analyze the data
                analysis_prompt = f"""You are a real estate data analyst. Based on the property data provided, answer the user's question with specific numbers and calculations.

Data: {stats_context}

User Question: {q}

Provide a detailed analysis with:
1. Direct answer to their question (with calculated numbers)
2. Key statistics (average, range, etc.)
3. Notable insights
4. Be specific and use the actual data provided."""

                if args.llm_engine != "none":
                    # For analytical queries with Gemini, optionally switch to pro model
                    if args.llm_engine == "gemini" and args.gemini_api_key:
                        answer_text = call_llm_gemini(analysis_prompt, args.gemini_analytical_model, args.gemini_api_key, 512)
                    else:
                        answer_text = call_llm_for_general_chat(analysis_prompt)
                    return answer_text
                else:
                    # Fallback: basic statistical analysis
                    prices = [p['price_numeric'] for p in all_properties if p['price_numeric'] > 0]
                    if prices:
                        avg_price = sum(prices) / len(prices)
                        min_price = min(prices)
                        max_price = max(prices)
                        answer_text = f"Based on {len(all_properties)} properties:\n"
                        answer_text += f"- Average price: PKR {avg_price:,.0f}\n"
                        answer_text += f"- Price range: PKR {min_price:,} to PKR {max_price:,}\n"
                        answer_text += f"- Total properties analyzed: {len(all_properties)}"
                        return answer_text
                    else:
                        return "No valid price data found for analysis."
            else:
                return "No properties found matching your criteria for analysis."
        
        # Regular property search (non-statistical)
        # Normalize location terms and extract numbers
        normalized_query = normalize_location_query(q)
        requested_number = extract_requested_number(q)
        
        # Check if this is a follow-up question that needs context expansion
        expanded_query = normalized_query
        if should_use_context(q, conversation_history):
            last_exchange = conversation_history[-1] if conversation_history else None
            if last_exchange:
                # Combine with previous query for better retrieval
                expanded_query = f"{last_exchange['user']} {normalized_query}"
        
        # Use enhanced retrieval with error handling
        res = enhanced_retrieve(args.collection, expanded_query, args.k, args.embedding_model, user_prefs, analysis.get('filters', {}))

        # Check if we found relevant property results
        if has_relevant_property_results(res):
            # Adjust top_n based on user request
            if requested_number:
                actual_top_n = min(requested_number, 5)  # Cap at 5 for performance
            else:
                # For general queries, let LLM decide from context but limit retrieval to 3 for better relevance
                actual_top_n = 3
            
            context, listings = build_context(res, processed_map, top_n=actual_top_n)
            
            # Apply strict property type filtering
            if analysis.get('filters', {}).get('property_type'):
                filtered_listings = []
                requested_type = analysis['filters']['property_type'].lower()
                
                for listing in listings:
                    title = listing.get('title', '').lower()
                    include_listing = True
                    
                    if requested_type == 'house':
                        # For house requests, exclude apartments/flats
                        if any(word in title for word in ['apartment', 'flat', 'unit']):
                            include_listing = False
                        # Must contain house-related terms
                        elif not any(word in title for word in ['house', 'home', 'villa', 'bungalow', 'marla']):
                            include_listing = False
                    elif requested_type == 'apartment':
                        # For apartment requests, exclude houses
                        if any(word in title for word in ['house', 'home', 'villa', 'bungalow', 'marla']):
                            include_listing = False
                        # Must contain apartment-related terms
                        elif not any(word in title for word in ['apartment', 'flat', 'unit']):
                            include_listing = False
                    
                    if include_listing:
                        filtered_listings.append(listing)
                
                listings = filtered_listings
                
                # If no listings match the strict filter, inform the user
                if not listings:
                    property_type_name = "houses" if requested_type == 'house' else "apartments/flats"
                    return f"No {property_type_name} found matching your criteria in the database. Try adjusting your search parameters."
            
            if args.explain:
                print("Retrieved chunks (debug):\n")
                print(context)
            
            # Enhanced system prompt that enforces strict adherence to context
            system = (
                "You are a conversational real-estate assistant. CRITICAL RULES:\n"
                "1. Use ONLY information from the context below - NO external knowledge\n"
                "2. Provide ONLY a brief conversational response - do NOT list property details\n"
                "3. Property listings will be shown separately in a formatted table\n"
                "4. If no relevant properties found, say 'No matching properties found in the database'\n"
                "5. If properties are found, say something like 'I found X properties matching your criteria'\n"
                "6. Do NOT format or list individual properties - just provide conversational context\n"
                "7. Each new query is independent - respond to what the user is currently asking\n"
                "8. Do NOT question or compare with previous requests - just answer the current query\n"
                "9. Keep your response brief and conversational\n"
                "10. If user asks for recommendations or similar properties, mention that feature\n"
                "11. STRICT PROPERTY TYPE FILTERING: Houses and apartments are completely separate - never mix them"
            )
            
            # Build conversation context - only for queries that truly need it
            conversation_context = ""
            if should_use_context(q, conversation_history):
                conversation_context = "\nRECENT CONTEXT (for reference only):\n"
                for exchange in conversation_history[-1:]:  # Only last exchange
                    conversation_context += f"Previous: {exchange['user']}\n"
                conversation_context += "END CONTEXT\n\n"
            
            # Add specific instruction about number if detected
            number_instruction = ""
            if requested_number:
                number_instruction = f"\nThe user asked for {requested_number} properties. Mention this in your response."
            
            # Add property type instruction
            property_type_instruction = ""
            if analysis.get('filters', {}).get('property_type'):
                prop_type = analysis['filters']['property_type']
                property_type_instruction = f"\nIMPORTANT: User specifically asked for {prop_type}s - results are strictly filtered for this property type only."
            
            prompt = (
                f"System: {system}{number_instruction}{property_type_instruction}\n\n"
                f"{conversation_context}"
                f"CONTEXT (use ONLY this information):\n{context}\n\n"
                f"User Query: {q}\n\n"
                f"INSTRUCTIONS:\n"
                f"- Provide ONLY a brief conversational response\n"
                f"- Do NOT list or format property details\n"
                f"- If properties found, just say how many match the criteria\n"
                f"- If no matches, say 'No matching properties found'\n"
                f"- Answer the current query directly - do NOT question the user's request\n"
                f"- Keep response under 2 sentences\n\n"
                "Brief Response:"
            )
            
            answer_text = ""
            if args.llm_engine == "llama" and args.llama_model_path:
                answer_text = call_llm_llamacpp(prompt, args.llama_model_path)
            elif args.llm_engine == "gpt4all":
                answer_text = call_llm_gpt4all(prompt, args.gpt4all_model)
            elif args.llm_engine == "ollama":
                answer_text = call_llm_ollama(prompt, args.ollama_model)
            elif args.llm_engine == "gemini" and args.gemini_api_key:
                # For non-analytical retrieval summaries flash model is enough
                answer_text = call_llm_gemini(prompt, args.gemini_model, args.gemini_api_key)
            else:
                answer_text = f"Found {len(listings)} matching properties" if listings else "No matching properties found"
            
            dates = [x.get("scraped_at") for x in listings if x.get("scraped_at")]
            if dates:
                freshness = f"Data scraped between {min(dates)} and {max(dates)}."
            else:
                freshness = "Data freshness unknown (scraped_at not available)."
            
            # Ensure we only show the exact number requested
            final_listings = listings[:actual_top_n] if listings else []
            
            # Validate response quality
            quality_report = validate_response_quality(q, answer_text, final_listings)
            
            # Add quality suggestions if score is low
            formatted_response = format_answer(q, final_listings, freshness, answer_text)
            
            # Check for recommendation requests
            if analysis['intent'] == 'recommend' and final_listings:
                # Get similar properties for the first result
                similar_props = get_similar_properties(final_listings[0]['listing_id'], processed_map, 2)
                if similar_props:
                    formatted_response += "\n\nSIMILAR PROPERTIES YOU MIGHT LIKE:\n"
                    formatted_response += "=" * 40 + "\n"
                    for i, prop in enumerate(similar_props, 1):
                        formatted_response += f"\n{i}. {prop.get('title', 'No title')}\n"
                        formatted_response += f"   Price: PKR {prop.get('price_numeric', 0):,}\n"
                        if prop.get('bedrooms'):
                            formatted_response += f"   Bedrooms: {prop['bedrooms']}\n"
            
            if quality_report['overall'] < 0.7 and quality_report['suggestions']:
                formatted_response += f"\n\nSuggestions: {'; '.join(quality_report['suggestions'])}"
            
            return formatted_response
        
        else:
            # No relevant property data found - forward to LLM for general conversation
            if args.explain:
                print("No relevant property data found. Forwarding to LLM for general conversation.\n")

            # Include conversation history in general chat
            conversation_context = ""
            if conversation_history:
                conversation_context = "Previous conversation:\n"
                for exchange in conversation_history[-3:]:  # Last 3 exchanges
                    conversation_context += f"User: {exchange['user']}\n"
                    conversation_context += f"Assistant: {exchange['assistant']}\n\n"
                conversation_context += "Current query:\n"

            general_prompt = (
                f"{conversation_context}You are a helpful AI assistant specializing in real estate. "
                f"The user asked: '{q}'. Please provide a helpful and friendly response considering the conversation context."
            )
            answer_text = call_llm_for_general_chat(general_prompt)
            return answer_text
            print("-" * 60)

    if args.interactive:
        print("=" * 60)
        print("REAL ESTATE RAG ASSISTANT - Interactive Chat")
        print("=" * 60)
        print("Ask me about properties in Bahria Town Phase 7!")
        print("Examples:")
        print("  - Show me 10 marla houses under 5 crore")
        print("  - Find apartments with 2 bedrooms")
        print("  - What's available near parks?")
        print("\nType 'exit', 'quit', or press Ctrl+C to quit.")
        print("-" * 60)
        
        # Initialize conversation history
        conversation_history = []
        
        try:
            while True:
                print()  # Add some spacing
                q = input("You: ").strip()
                if not q:
                    continue
                if q.lower() in {"exit", "quit", "bye", "goodbye"}:
                    print("\nGoodbye! Happy house hunting!")
                    break
                
                print("Assistant: Searching...")
                response = answer_one(q, conversation_history)
                print(f"\n{response}")
                
                # Add to conversation history
                conversation_history.append({"user": q, "assistant": response})
                
                print("-" * 60)
                
        except KeyboardInterrupt:
            print("\n\nGoodbye! Happy house hunting!")
        return 0
    else:
        if not args.query:
            print("--query is required unless --interactive is used")
            return 1
        answer_one(args.query)
        return 0


if __name__ == "__main__":
    raise SystemExit(main())

