import os
import sys
import importlib.util

# Resolve base paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SUBDIR = os.path.join(BASE_DIR, "NCAFCompare")

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
    """Fallback: load Flask app from the NCAFCompare subfolder without requiring package import."""
    app_path = os.path.join(SUBDIR, "app.py")
    if os.path.exists(app_path):
        spec = importlib.util.spec_from_file_location("ncaaf_app_sub", app_path)
        if spec and spec.loader:
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)  # type: ignore[attr-defined]
            if hasattr(mod, "app"):
                return getattr(mod, "app")
    return None

# Try root app first, then subfolder, then package import
application = _load_root_app()
if application is None:
    # Add subdir to path so relative imports inside it work, then try direct path load
    if os.path.isdir(SUBDIR) and SUBDIR not in sys.path:
        sys.path.insert(0, SUBDIR)
    application = _load_subfolder_app()

if application is None:
    # Final fallback to package import if available
    from NCAFCompare.app import app as application  # type: ignore

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5051))
    application.run(host="0.0.0.0", port=port, debug=False)
