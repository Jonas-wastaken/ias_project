import os
import logging

# Create logs directory if it doesn't exist
log_dir = os.path.join("tests", "logs")
os.makedirs(log_dir, exist_ok=True)


def setup_logging(test_name):
    """Sets up and returns a separate logger for each test file, ensuring logs are not printed to console."""
    logger = logging.getLogger(test_name)  # Create a unique logger per test
    logger.setLevel(logging.INFO)  # Set logging level

    # Remove existing handlers to prevent duplicate logs (console handlers)
    if logger.hasHandlers():
        logger.handlers.clear()

    # Define log file path
    log_file = os.path.join(log_dir, f"{test_name}.log")

    # Create a file handler for logging
    file_handler = logging.FileHandler(log_file, mode="w")
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")
    )

    # Add the file handler (no console handler)
    logger.addHandler(file_handler)

    return logger  # Return the configured logger
