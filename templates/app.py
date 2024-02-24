import os
from flask_pymongo import PyMongo

from flask import Flask, flash, redirect, render_template, request, session
from flask_session import Session
from werkzeug.security import check_password_hash, generate_password_hash


from helpers import apology, login_required, lookup, usd

# Configure application
app = Flask(__name__)
app.config["TEMPLATES_AUTO_RELOAD"] = True

# Initialize PyMongo with your MongoDB URI
app.config['MONGO_URI'] = 'mongodb://localhost:27017/your_database_name'
db = PyMongo(app)

# Custom filter
@app.template_filter('usd')
def usd(value):
    return f"${value:,.2f}"

@app.after_request
def after_request(response):
    """Ensure responses aren't cached"""
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Expires"] = 0
    response.headers["Pragma"] = "no-cache"
    return response


@app.route("/")
@login_required
def index():
    """Show portfolio of stocks"""

    return render_template("index.html")


@app.route("/buy", methods=["GET", "POST"])
@login_required
def buy():
    """Buy shares of stock"""

    # define BUY action
    action = "BUY"

    if request.method == "POST":
        symbol = request.form.get("symbol")
        shares = int(request.form.get("shares"))

        # if symbol or shares are empty
        if not symbol or not shares:
            flash("Stock symbol and number of shares must not be empty.", "error")
        # if user manually changed shares to 0 or negative int
        elif shares <= 0:
            flash("Number of shares must be a positive integer.", "error")
        else:
            quote = lookup(symbol)
            # if quote is null
            if not quote:
                flash("Stock symbol does not exist.", "error")
            else:
                user_id = session["user_id"]
                price = quote["price"]
                name = quote["name"]
                # check cash balance
                user = db.users.find_one({"_id": user_id})
                cash = user["cash"]

                # check stock purchase total cost
                cost = price * shares

                # balance has to be >= 0
                balance = cash - cost

                # if total cost exceeds cash balance, return error message
                if balance < 0:
                    flash(f"You do not have enough cash balance. Current balance: {usd(cash)}. Total cost to buy stock: {usd(cost)}", "error")
                else:
                    # insert buy transaction
                    db.transactions.insert_one({
                        "user_id": user_id,
                        "action": action,
                        "symbol": symbol,
                        "name": name,
                        "price": price,
                        "shares": shares
                    })

                    # update user document with new cash balance
                    db.users.update_one({"_id": user_id}, {"$set": {"cash": balance}})

                    flash("Stock bought!", "information")
                    return redirect("/")

    return render_template("buy.html", action=action)

@app.route("/history")
@login_required
def history():
    """Show history of transactions"""

    user_id = session["user_id"]
    # Get this user's transactions
    trades = db.transactions.find({"user_id": user_id}).sort([("symbol", 1), ("datetime", 1)])

    return render_template("history.html", trades=trades)

@app.route("/login", methods=["GET", "POST"])
def login():
    """Log user in"""

    # Forget any user_id
    session.clear()

    # User reached route via POST (as by submitting a form via POST)
    if request.method == "POST":

        # Ensure username was submitted
        if not request.form.get("username"):
            flash("Must provide username", "error")
            return redirect("/login")

        # Ensure password was submitted
        elif not request.form.get("password"):
            flash("Must provide password", "error")
            return redirect("/login")

        # Query database for username
        user = db.users.find_one({"username": request.form.get("username")})

        # Ensure username exists and password is correct
        if user is None or not check_password_hash(user["hash"], request.form.get("password")):
            flash("Invalid username and/or password", "error")
            return redirect("/login")

        # Remember which user has logged in
        session["user_id"] = user["_id"]

        # Display message
        flash("Logged in!", "information")

        # Redirect user to home page
        return redirect("/")

    # User reached route via GET (as by clicking a link or via redirect)
    else:
        return render_template("login.html")

@app.route("/logout")
def logout():
    """Log user out"""

    # Forget any user_id
    session.clear()

    # Redirect user to login form
    return redirect("/login")

@app.route("/register", methods=["GET", "POST"])
def register():
    """Register user"""

    session.clear()

    if request.method == "POST":
        # Get user inputs
        username = request.form.get("username")
        password = request.form.get("password")
        confirmation = request.form.get("confirmation")

        if not username:
            return apology("Username must not be empty.")

        # Check if username already exists
        user = db.users.find_one({"username": username})
        if user:
            return apology("Username already exists.")

        if not password or not confirmation:
            return apology("Password and confirmation must not be empty.")

        if password != confirmation:
            return apology("Passwords do not match")

        # Hash the password
        hash_pw = generate_password_hash(password)

        # Register the new user
        db.users.insert_one({
            "username": username,
            "hash": hash_pw
        })

        # Display a message that the user is registered
        flash("User registered!", "information")

        return redirect("/login")

    # If the request method is GET
    return render_template("register.html")

@app.route("/sell", methods=["GET", "POST"])
@login_required
def sell():
    """Sell shares of stock"""

    # Define SELL action
    action = "SELL"
    user_id = session["user_id"]

    # Get user stocks information
    stocks = list(db.transactions.aggregate([
        {"$match": {
            "user_id": user_id,
            "symbol": {"$ne": ""},
            "action": "BUY"
        }},
        {"$group": {
            "_id": "$symbol",
            "name": {"$first": "$name"},
            "shares": {"$sum": "$shares"}
        }}
    ]))

    # GET
    if request.method != "POST":
        return render_template("sell.html", stocks=stocks)

    # POST
    symbol = request.form.get("symbol")
    shares = request.form.get("shares", type=int)

    # If symbol or shares are empty
    if not symbol or not shares:
        return apology("Stock symbol and number of shares must not be empty.")

    # If the number of shares is not a positive integer
    if shares <= 0:
        return apology("Number of shares must be a positive integer")

    shares_owned = 0
    for stock in stocks:
        if stock["_id"] == symbol:
            shares_owned = stock["shares"]

    if shares > shares_owned:
        return apology("You do not own that many shares of this stock")

    quote = lookup(symbol)
    price = quote["price"]
    name = quote["name"]

    # Insert sell transaction
    db.transactions.insert_one({
        "user_id": user_id,
        "action": action,
        "symbol": symbol,
        "name": name,
        "price": price,
        "shares": shares
    })

    # Update user cash balance
    user = db.users.find_one({"_id": user_id})
    cash = user["cash"]
    balance = cash + price * shares

    db.users.update_one({"_id": user_id}, {"$set": {"cash": balance}})

    # Display message that stock sold
    flash("Stock sold!", "information")

    return redirect("/")

if __name__ == '__main__':
    app.run(debug=True)