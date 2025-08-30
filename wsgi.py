import os
import sys
import importlib.util

# Load Flask app from the NCAFCompare subfolder without requiring package import
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SUBDIR = os.path.join(BASE_DIR, "NCAFCompare")
if os.path.isdir(SUBDIR) and SUBDIR not in sys.path:
    sys.path.insert(0, SUBDIR)

def _load_subfolder_app():
    app_path = os.path.join(SUBDIR, "app.py")
    if os.path.exists(app_path):
        spec = importlib.util.spec_from_file_location("ncaaf_app_sub", app_path)
        if spec and spec.loader:
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)  # type: ignore[attr-defined]
            if hasattr(mod, "app"):
                return getattr(mod, "app")
    return None

application = _load_subfolder_app()
if application is None:
    # Fallback to package import if available
    from NCAFCompare.app import app as application  # type: ignore

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5051))
    application.run(host="0.0.0.0", port=port, debug=False)
