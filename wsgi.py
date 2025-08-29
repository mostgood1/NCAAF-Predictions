import os
from NCAFCompare.app import app as application; fallback to from app import app as application  # Expose as 'application' for WSGI servers

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5051))
    application.run(host="0.0.0.0", port=port, debug=False)
