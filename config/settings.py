"""
Configuration settings for NCAA Football Prediction application.
"""

import os
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent

class Config:
    """Base configuration class."""
    
    # Flask configuration
    SECRET_KEY = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')
    DEBUG = os.environ.get('DEBUG', 'True').lower() == 'true'
    HOST = os.environ.get('HOST', '127.0.0.1')
    PORT = int(os.environ.get('PORT', 5000))
    
    # Data directories
    DATA_DIR = PROJECT_ROOT / 'data'
    RAW_DATA_DIR = DATA_DIR / 'raw'
    PROCESSED_DATA_DIR = DATA_DIR / 'processed'
    MODELS_DIR = DATA_DIR / 'models'
    
    # Model configuration
    MODEL_RANDOM_STATE = 42
    TEST_SIZE = 0.2
    CROSS_VALIDATION_FOLDS = 5
    
    # Data collection settings
    API_TIMEOUT = 30
    SCRAPING_DELAY = 1  # seconds between requests
    USER_AGENT = 'NCAA-Football-Predictor/1.0'
    
    # Database settings (if using database)
    DATABASE_URL = os.environ.get('DATABASE_URL', 'sqlite:///ncaa_football.db')
    
    # API Keys (set these as environment variables)
    SPORTS_API_KEY = os.environ.get('SPORTS_API_KEY')
    WEATHER_API_KEY = os.environ.get('WEATHER_API_KEY')
    
    # Prediction settings
    CONFIDENCE_THRESHOLD = 0.6
    MINIMUM_GAMES_FOR_PREDICTION = 5
    
    # Feature engineering settings
    LOOKBACK_GAMES = 10  # Number of previous games to consider
    WEIGHT_DECAY = 0.95  # How much to weight recent vs older games
    
    @classmethod
    def ensure_directories(cls):
        """Create necessary directories if they don't exist."""
        directories = [
            cls.DATA_DIR,
            cls.RAW_DATA_DIR,
            cls.PROCESSED_DATA_DIR,
            cls.MODELS_DIR
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

class DevelopmentConfig(Config):
    """Development configuration."""
    DEBUG = True
    
class ProductionConfig(Config):
    """Production configuration."""
    DEBUG = False
    SECRET_KEY = os.environ.get('SECRET_KEY')
    
    if not SECRET_KEY:
        raise ValueError("SECRET_KEY environment variable must be set in production")

class TestingConfig(Config):
    """Testing configuration."""
    TESTING = True
    DEBUG = True
    DATABASE_URL = 'sqlite:///:memory:'

# Configuration mapping
config_map = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}

def get_config(config_name=None):
    """Get configuration class based on environment."""
    if config_name is None:
        config_name = os.environ.get('FLASK_ENV', 'default')
    
    return config_map.get(config_name, DevelopmentConfig)
