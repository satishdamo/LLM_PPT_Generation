from app import app
from waitress import serve
import logging

if __name__ == "__main__":
    # Configure logging for the application
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger("waitress")
    logger.info("Starting Waitress server on http://0.0.0.0:5000")

    # Start the Waitress server
    serve(app, host="0.0.0.0", port=5000, threads=4)
