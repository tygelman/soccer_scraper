import os

class Config:
    SECRET_KEY = os.environ.get('FLASK_SECRET', 'this-is-sooooo-secret')
    DEBUG = True
    FLASK_APP = 'app.py'
    FLASK_ENV = 'development'