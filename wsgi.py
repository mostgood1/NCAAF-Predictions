import os
import sys
import importlib.util
import traceback
from typing import Optional
try:
    from flask import Flask, Response
except Exception:  # Flask will be installed in Render
    Flask = None  # type: ignore
    Response = None  # type: ignore

# Resolve base paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SUBDIR = os.path.join(BASE_DIR, "NCAFCompare")
FORCE_ROOT_ONLY = True  # prevent falling back to stale sub-app

def _load_root_app():
    """Prefer loading the robust root-level app.py (app: Flask)."""
    app_path = os.path.join(BASE_DIR, "app.py")
    if os.path.exists(app_path):
        spec = importlib.util.spec_from_file_location("root_app_module", app_path)
        if spec and spec.loader:
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)  # type: ignore[attr-defined]
            if hasattr(mod, "app"):
                return getattr(mod, "app")
    # Try standard import if on sys.path
    try:
        from app import app as root_application  # type: ignore
        return root_application
    except Exception:
        return None

def _load_subfolder_app():
    if FORCE_ROOT_ONLY:
        return None
    # legacy path retained for reference if FORCE_ROOT_ONLY is False
    if os.path.isdir(SUBDIR):
        if SUBDIR not in sys.path:
            sys.path.insert(0, SUBDIR)
        app_path = os.path.join(SUBDIR, "app.py")
        if os.path.exists(app_path):
            spec = importlib.util.spec_from_file_location("ncaaf_app_sub", app_path)
            if spec and spec.loader:
                mod = importlib.module_from_spec(spec)
                spec.loader.exec_module(mod)  # type: ignore[attr-defined]
                if hasattr(mod, "app"):
                    return getattr(mod, "app")
    return None

startup_error: Optional[str] = None
application = None

# Try root app first, then subfolder, then package import, capturing errors
try:
    application = _load_root_app()
    if application and hasattr(application, '__class__'):
        try:
            print(f"[wsgi] Loaded root app: {getattr(application, '__module__', '?')}")
        except Exception:
            pass
except Exception:
    startup_error = traceback.format_exc()

if application is None and not FORCE_ROOT_ONLY:
    try:
        application = _load_subfolder_app()
    except Exception:
        if not startup_error:
            startup_error = traceback.format_exc()

if application is None and not FORCE_ROOT_ONLY:
    try:
        from NCAFCompare.app import app as application  # type: ignore
    except Exception:
        if not startup_error:
            startup_error = traceback.format_exc()
        application = None

# If still not available or an error occurred, provide a minimal fallback app so we don't 502
if application is None or startup_error is not None:
    if Flask is None:
        # Last-ditch placeholder WSGI callable
        def application(environ, start_response):  # type: ignore
            start_response('500 INTERNAL SERVER ERROR', [('Content-Type', 'text/plain')])
            body = startup_error.encode('utf-8') if startup_error else b'Application failed to start.'
            return [body]
    else:
        _fallback = Flask(__name__)

        @_fallback.route('/')
        def _root():
            msg = 'Application failed to start. Visit /startup-error for details.' if startup_error else 'Application not found.'
            return msg, 500

        @_fallback.route('/startup-error')
        def _err():
            text = startup_error or 'No error captured.'
            return Response(text, mimetype='text/plain')

        @_fallback.route('/health')
        def _health():
            return {'status': 'error', 'message': 'startup_failed', 'has_trace': bool(startup_error)}, 500

        application = _fallback
else:
    # Inject a small diagnostics route into the loaded Flask app if possible
    try:
        if hasattr(application, 'add_url_rule'):
            def _which():
                return {
                    'loaded_from_root': True,
                    'force_root_only': FORCE_ROOT_ONLY,
                    'base_dir': BASE_DIR,
                    'has_startup_error': bool(startup_error),
                    'module': getattr(application, '__module__', 'unknown')
                }
            # avoid duplicate rule errors
            existing = [r.rule for r in getattr(application, 'url_map').iter_rules()]  # type: ignore[attr-defined]
            if '/which-app' not in existing:
                application.add_url_rule('/which-app', 'which_app', _which)  # type: ignore[arg-type]
    except Exception:
        pass

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5051))
    application.run(host="0.0.0.0", port=port, debug=False)
