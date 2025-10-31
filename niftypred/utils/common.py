"""
Utility functions for timing and logging
"""
import time
from functools import wraps
import logging
from datetime import datetime

# Configure logging with UTF-8 encoding
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('nifty_predictor')

class Timer:
    def __init__(self, name=None):
        self.name = name
        self.start_time = None
        
    def __enter__(self):
        self.start_time = time.time()
        return self
        
    def __exit__(self, *args):
        elapsed_time = time.time() - self.start_time
        if self.name:
            logger.info(f"{self.name} took {elapsed_time:.2f} seconds")
        return False

def log_execution_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed_time = time.time() - start_time
        logger.info(f"{func.__name__} executed in {elapsed_time:.2f} seconds")
        return result, elapsed_time
    return wrapper