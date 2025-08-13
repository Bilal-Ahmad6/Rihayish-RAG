import os
import sys
import json
from pathlib import Path
from datetime import datetime
from threading import Thread
from typing import List, Dict, Any

from flask import Flask, render_template, request, jsonify
import logging
from logging.handlers import RotatingFileHandler

# Load .env early
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv(dotenv_path=Path(__file__).parent.parent / '.env')
except Exception:
    pass

# Add project root to path so we can import scripts.query_rag
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'scripts'))
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.query_rag import rag_infer, load_processed_map  # type: ignore
from config import settings  # type: ignore

app = Flask(__name__)


# IMPORTANT: Do NOT run Flask's built-in dev server in production. Always use Gunicorn with PORT env var.
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'dev-key-change-in-production')

# Log the port for Render debugging
_render_port = os.getenv('PORT')
if _render_port:
    # Check if PORT is a template variable that wasn't resolved
    if _render_port.startswith("${") or _render_port.startswith("$"):
        app.logger.warning(f"[Render] PORT environment variable appears to be unresolved template: {_render_port}")
    else:
        app.logger.info(f"[Render] App will be served on PORT={_render_port} (set by Render)")
else:
    app.logger.warning("[Render] PORT environment variable is not set! App may not be reachable by Render health checks.")

# Configure logging for production
if not app.debug:
    # Ensure logs directory exists
    logs_dir = PROJECT_ROOT / 'logs'
    logs_dir.mkdir(exist_ok=True)
    
    # Set up rotating file handler
    file_handler = RotatingFileHandler(
        logs_dir / 'app.log', 
        maxBytes=10240000, 
        backupCount=10
    )
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
    ))
    file_handler.setLevel(logging.INFO)
    app.logger.addHandler(file_handler)
    app.logger.setLevel(logging.INFO)
    app.logger.info('PropertyGuru application startup')

# In-memory + file-backed conversation store
CONVERSATIONS: Dict[str, List[Dict[str, Any]]] = {}
PERSIST_PATH = PROJECT_ROOT / 'web_ui' / 'conversations.json'


def load_conversations() -> None:
    if PERSIST_PATH.exists():
        try:
            data = json.loads(PERSIST_PATH.read_text(encoding='utf-8'))
            if isinstance(data, dict):
                # Basic validation
                for k, v in data.items():
                    if isinstance(v, list):
                        CONVERSATIONS[k] = v
        except Exception:
            pass


def save_conversations() -> None:
    try:
        PERSIST_PATH.write_text(json.dumps(CONVERSATIONS, ensure_ascii=False, indent=2), encoding='utf-8')
    except Exception:
        pass


load_conversations()


def get_api_key() -> str:
    return os.getenv('GEMINI_API_KEY') or os.getenv('GRAANA_GEMINI_API_KEY') or settings.gemini_api_key


def _background_warmup() -> None:
    """Warm up embedding model to avoid cold-start timeouts in deployment."""
    try:
        app.logger.info('Warmup: starting model download/init...')
        # Prefer the model used by query path
        try:
            # Import lazily to avoid overhead if unused
            from scripts.query_rag import embed_query  # type: ignore
            _ = embed_query('sentence-transformers/all-mpnet-base-v2', 'warmup')
            app.logger.info('Warmup: mpnet embed initialized')
        except Exception as e:
            app.logger.warning(f'Warmup: mpnet init failed: {e}')
        # Also try configured model (may be MiniLM)
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
            SentenceTransformer(getattr(settings, 'embedding_model', 'sentence-transformers/all-mpnet-base-v2'))
            app.logger.info('Warmup: settings.embedding_model initialized')
        except Exception as e:
            app.logger.warning(f'Warmup: settings model init failed: {e}')
        app.logger.info('Warmup: completed')
    except Exception as e:
        app.logger.warning(f'Warmup encountered an error: {e}')


# Kick off warmup in background (non-blocking)
try:
    Thread(target=_background_warmup, daemon=True).start()
except Exception:
    pass


@app.errorhandler(404)
def not_found_error(error):
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    app.logger.error(f'Server Error: {error}')
    return jsonify({'error': 'Internal server error'}), 500


@app.errorhandler(Exception)
def handle_exception(e):
    app.logger.error(f'Unhandled exception: {e}')
    return jsonify({'error': 'An unexpected error occurred'}), 500


@app.route('/health')
def health_check():
    """Health check endpoint for deployment monitoring"""
    try:
        # Basic health checks
        api_key = get_api_key()
        chroma_exists = (PROJECT_ROOT / 'chromadb_data').exists()
        
        status = {
            'status': 'healthy',
            'api_key_configured': bool(api_key),
            'vector_db_exists': chroma_exists,
            'conversations_loaded': len(CONVERSATIONS),
            'timestamp': datetime.now().isoformat()
        }
        return jsonify(status), 200
    except Exception as e:
        # Ensure health check always returns something
        error_status = {
            'status': 'error',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }
        return jsonify(error_status), 500


@app.route('/live')
def liveness():
    """Lightweight liveness probe (no heavy dependencies)."""
    return jsonify({'status': 'alive', 'timestamp': datetime.utcnow().isoformat()}), 200


_READINESS_CACHE = {
    'ok': False,
    'last_check': None,
    'details': {},
    'error': None
}


@app.route('/ready')
def readiness():
    """Readiness probe performing deeper checks (vector DB + optional model import).
    Cached briefly to avoid heavy repeated work if platform pings frequently.
    """
    from time import time
    now = time()
    try:
        # Reuse cached result if recent (< 30s)
        if _READINESS_CACHE['last_check'] and (now - _READINESS_CACHE['last_check'] < 30):
            code = 200 if _READINESS_CACHE['ok'] else 503
            return jsonify({**_READINESS_CACHE, 'cached': True}), code

        details = {}
        # API key check
        details['api_key_present'] = bool(get_api_key())

        # Vector DB / collection check
        chroma_dir = PROJECT_ROOT / 'chromadb_data'
        details['vector_path'] = str(chroma_dir)
        if chroma_dir.exists():
            try:
                import chromadb  # type: ignore
                client = chromadb.PersistentClient(path=str(chroma_dir))
                coll = client.get_or_create_collection(name=settings.collection_name)
                count = None
                if hasattr(coll, 'count'):
                    count = coll.count()
                details['vector_collection'] = settings.collection_name
                details['vector_count'] = count
                details['vector_ok'] = True
            except Exception as ve:  # pragma: no cover
                details['vector_ok'] = False
                details['vector_error'] = str(ve)
        else:
            details['vector_ok'] = False
            details['vector_error'] = 'directory_missing'

        # Optional embedding model import check (quick import, not full load if cached by sentence-transformers)
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
            # Do not load model fully if already in cache; attempt a fast init with minimal risk
            _ = SentenceTransformer(getattr(settings, 'embedding_model', 'sentence-transformers/all-MiniLM-L6-v2'))
            details['embedding_model_ok'] = True
        except Exception as me:  # pragma: no cover
            details['embedding_model_ok'] = False
            details['embedding_model_error'] = str(me)

        overall_ok = all([
            details.get('api_key_present'),
            details.get('vector_ok'),
            details.get('embedding_model_ok')
        ])

        _READINESS_CACHE.update({
            'ok': overall_ok,
            'last_check': now,
            'details': details,
            'error': None if overall_ok else 'one_or_more_checks_failed'
        })
        return jsonify(_READINESS_CACHE), (200 if overall_ok else 503)
    except Exception as e:  # pragma: no cover
        _READINESS_CACHE.update({
            'ok': False,
            'last_check': now,
            'details': {},
            'error': str(e)
        })
        return jsonify(_READINESS_CACHE), 503


@app.route('/api/debug/status')
def debug_status():
    """Lightweight diagnostics to verify data files and vector DB access."""
    try:
        # Processed data
        processed_path = PROJECT_ROOT / 'data' / 'processed' / 'graana_phase8_processed.json'
        processed_exists = processed_path.exists()
        processed_count = 0
        processed_error = None
        if processed_exists:
            try:
                pmap = load_processed_map(processed_path)
                processed_count = len(pmap)
            except Exception as e:
                processed_error = str(e)
        else:
            processed_error = 'processed file missing'

        # Vector DB
        chroma_dir = PROJECT_ROOT / 'chromadb_data'
        vector_db_exists = chroma_dir.exists()
        collection_count = None
        vector_error = None
        if vector_db_exists:
            try:
                import chromadb  # type: ignore
                client = chromadb.PersistentClient(path=str(chroma_dir))
                coll = client.get_or_create_collection(name=settings.collection_name)
                if hasattr(coll, 'count'):
                    collection_count = coll.count()
            except Exception as e:
                vector_error = str(e)

        out = {
            'status': 'ok',
            'processed': {
                'path': str(processed_path),
                'exists': processed_exists,
                'count': processed_count,
                'error': processed_error
            },
            'vector_db': {
                'path': str(chroma_dir),
                'exists': vector_db_exists,
                'collection': settings.collection_name,
                'count': collection_count,
                'error': vector_error
            },
            'timestamp': datetime.now().isoformat()
        }
        return jsonify(out), 200
    except Exception as e:
        return jsonify({'status': 'error', 'error': str(e)}), 500


@app.route('/')
def index():
    # Clear all conversations when the page is refreshed/accessed
    CONVERSATIONS.clear()
    save_conversations()
    app.logger.info('Conversations cleared on page refresh')
    return render_template('chat.html')


@app.route('/api/new_chat', methods=['POST'])
def new_chat():
    chat_id = datetime.utcnow().strftime('%Y%m%d%H%M%S%f')
    CONVERSATIONS[chat_id] = []
    save_conversations()
    return jsonify({"chat_id": chat_id})


@app.route('/api/list_chats')
def list_chats():
    # Return last 50 chats with first user message as title or default
    items = []
    for cid, msgs in sorted(CONVERSATIONS.items(), reverse=True):
        title = 'New Conversation'
        for m in msgs:
            if m.get('role') == 'user' and m.get('content'):
                raw = m['content'].strip().split('\n')[0]
                title = (raw[:40] + '...') if len(raw) > 40 else raw
                break
        items.append({'chat_id': cid, 'title': title})
    return jsonify({'chats': items[:50]})


@app.route('/api/get_chat/<chat_id>')
def get_chat(chat_id: str):
    return jsonify({'messages': CONVERSATIONS.get(chat_id, [])})


@app.route('/api/clear_history', methods=['POST'])
def clear_history():
    CONVERSATIONS.clear()
    save_conversations()
    return jsonify({'status': 'cleared'})


@app.route('/api/message', methods=['POST'])
def message():
    data = request.get_json(force=True)
    chat_id = data.get('chat_id')
    user_message = (data.get('message') or '').strip()
    store_history = bool(data.get('store_history', True))
    if not user_message:
        return jsonify({'error': 'Empty message'}), 400

    if not chat_id:
        # auto create
        chat_id = datetime.utcnow().strftime('%Y%m%d%H%M%S%f')
        CONVERSATIONS[chat_id] = []

    history = CONVERSATIONS.setdefault(chat_id, [])
    user_entry = {'role': 'user', 'content': user_message, 'ts': datetime.utcnow().isoformat()}
    history.append(user_entry)

    api_key = get_api_key()
    meta: Dict[str, Any] = {}
    if not api_key:
        assistant_text = 'Gemini API key missing. Please set GRAANA_GEMINI_API_KEY or GEMINI_API_KEY in your .env.'
        mode = 'error'
    else:
        # Build conversation_history structure for rag_infer
        # Limit conversation history to last 5 exchanges to avoid context overflow
        recent_history = history[-10:] if len(history) > 10 else history
        
        convo_for_rag = []
        for m in recent_history:
            if m['role'] == 'user':
                convo_for_rag.append({'user': m['content'], 'assistant': ''})
            elif m['role'] == 'assistant' and convo_for_rag:
                convo_for_rag[-1]['assistant'] = m['content']
        try:
            # Use Gemini as the LLM engine
            result = rag_infer(
                query=user_message,
                gemini_api_key=api_key,
                gemini_model="gemini-1.5-flash",
                gemini_analytical_model="gemini-1.5-pro",
                conversation_history=convo_for_rag
            )
            assistant_text = result.get('answer', 'No response generated.')
            mode = result.get('mode')
            
            # Handle new response format with listings
            if result.get('listings'):
                meta['listings'] = result['listings']
            # Forward price statistics if present
            if result.get('price_stats'):
                meta['price_stats'] = result['price_stats']
            
            # Attach extra metadata for compatibility
            meta.update({k: result.get(k) for k in ['mode', 'total_found', 'filters_applied'] if k in result})

        except Exception as e:
            # Special guidance for common errors
            err_str = str(e)
            if 'no such column: collections.topic' in err_str.lower():
                assistant_text = (
                    "Error accessing vector DB (schema mismatch). Delete the 'chromadb_data' folder and re-run embeddings (scripts/embed_and_store.py)."
                )
            elif 'request too large' in err_str.lower() or 'context length' in err_str.lower():
                assistant_text = (
                    "The request was too large for the model. Try asking a shorter, more specific question. "
                    "For example: 'Find 3 bedroom houses' instead of very long queries."
                )
            elif 'gemini' in err_str.lower():
                assistant_text = (
                    "There was an issue with the Gemini API. Please try again shortly."
                )
            else:
                assistant_text = f'Error: {e}'
            mode = 'error'
            meta['error'] = err_str
    assistant_entry = {'role': 'assistant', 'content': assistant_text, 'ts': datetime.utcnow().isoformat(), 'meta': meta}
    history.append(assistant_entry)

    if store_history:
        save_conversations()
    else:
        # If user disabled history, remove this chat entirely when empty
        pass

    return jsonify({'chat_id': chat_id, 'assistant_message': assistant_entry, 'messages': history, 'title': history[0]['content'] if history and history[0]['role']=='user' else 'New Conversation'})

if __name__ == "__main__":
    # Only run the Flask dev server if not running under Gunicorn
    if "GUNICORN_CMD_ARGS" not in os.environ:
        # Robust PORT handling for deployment environments
        port_env = os.environ.get("PORT", "10000")
        
        # Handle cases where PORT might be set to "${PORT}" or other invalid values
        try:
            # Remove common template patterns that might not be resolved
            if port_env.startswith("${") and port_env.endswith("}"):
                port_env = "10000"  # Fallback to default
            elif port_env.startswith("$"):
                port_env = "10000"  # Fallback to default
            
            port = int(port_env)
            
            # Validate port range
            if port < 1 or port > 65535:
                port = 10000
                
        except (ValueError, TypeError):
            # If conversion fails, use default port
            port = 10000
            app.logger.warning(f"Invalid PORT value '{port_env}', using default port 10000")
        
        app.logger.info(f"Starting Flask app on 0.0.0.0:{port}")
        app.run(host="0.0.0.0", port=port, debug=True)
