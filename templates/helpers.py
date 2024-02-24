import os
import requests
import urllib.parse

from flask import redirect, render_template, request, session
from functools import wraps
from functools import wraps
from flask import session, redirect, url_for

# Import MongoClient from PyMongo
from pymongo import MongoClient

# Create a MongoClient to connect to your MongoDB instance
client = MongoClient('mongodb://localhost:27017/')
db = client['your_database_name']

def apology(message, code=400):
    """Render message as an apology to the user."""
    def escape(s):
        """
        Escape special characters.

        https://github.com/jacebrowning/memegen#special-characters
        """
        for old, new in [("-", "--"), (" ", "-"), ("_", "__"), ("?", "~q"),
                         ("%", "~p"), ("#", "~h"), ("/", "~s"), ("\"", "''")]:
            s = s.replace(old, new)
        return s
    
    # You can query MongoDB here to retrieve any necessary data
    # Example: error_data = db.errors.find_one({"error_code": code})
    # This assumes you have a MongoDB collection named "errors" with error messages
    
    # Pass the error_data to the render_template function if you need specific error messages
    
    return render_template("apology.html", top=code, bottom=escape(message)), code

def login_required(f):
    """
    Decorate routes to require login.
    """
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' in session:
            # Get the user data from MongoDB
            user_id = session['user_id']
            user = db.users.find_one({'_id': user_id})
            
            if user:
                # User is authenticated, proceed with the route
                return f(user, *args, **kwargs)
        
        # If user is not authenticated, redirect to the login page
        return redirect(url_for('login'))

    return decorated_function

def usd(value):
    """Format value as USD."""
    return f"${value:,.2f}"
