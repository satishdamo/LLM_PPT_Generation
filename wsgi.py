from app import app
from waitress import serve
import logging

if __name__ == "__main__":
    # Start the Waitress server
    serve(app, host="0.0.0.0", port=5000)
