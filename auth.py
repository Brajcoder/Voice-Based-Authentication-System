"""
Authentication blueprint for Voice-Based Authentication System.
Handles registration, password-based login, logout, and protected dashboard access.
"""

import functools
import bcrypt
from flask import (
    Blueprint,
    render_template,
    request,
    redirect,
    url_for,
    session,
    flash,
    jsonify,
)
from database import (
    create_user,
    get_user_by_username,
    get_user_by_id,
)

auth_bp = Blueprint("auth", __name__)


def hash_password(password: str) -> str:
    """Hash password using bcrypt."""
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(password.encode("utf-8"), salt).decode("utf-8")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify plain password against bcrypt hash."""
    try:
        return bcrypt.checkpw(
            plain_password.encode("utf-8"), hashed_password.encode("utf-8")
        )
    except Exception:
        return False


def login_required(view):
    """Decorator to protect routes requiring authentication."""

    @functools.wraps(view)
    def wrapped_view(**kwargs):
        if "user_id" not in session:
            flash("Please log in to access this page.", "warning")
            return redirect(url_for("auth.login"))
        return view(**kwargs)

    return wrapped_view


@auth_bp.route("/register", methods=["GET", "POST"])
def register():
    """User registration route."""
    if "user_id" in session:
        return redirect(url_for("auth.dashboard"))

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        confirm_password = request.form.get("confirm_password", "")

        # Validation
        if not username or len(username) < 3:
            flash("Username must be at least 3 characters long.", "danger")
            return render_template("register.html", username=username)

        if not password or len(password) < 6:
            flash("Password must be at least 6 characters long.", "danger")
            return render_template("register.html", username=username)

        if password != confirm_password:
            flash("Passwords do not match.", "danger")
            return render_template("register.html", username=username)

        pwd_hash = hash_password(password)
        user_id = create_user(username, pwd_hash)

        if user_id is None:
            flash("Username is already taken. Please choose another.", "danger")
            return render_template("register.html", username=username)

        # Log in the newly registered user
        session.clear()
        session["user_id"] = user_id
        session["username"] = username
        flash("Registration successful! You can now enroll your voice signature.", "success")
        return redirect(url_for("auth.dashboard"))

    return render_template("register.html")


@auth_bp.route("/login", methods=["GET", "POST"])
def login():
    """User login route."""
    if "user_id" in session:
        return redirect(url_for("auth.dashboard"))

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")

        if not username or not password:
            flash("Please enter both username and password.", "danger")
            return render_template("index.html", username=username)

        user = get_user_by_username(username)
        if user is None or not verify_password(password, user["password_hash"]):
            flash("Invalid username or password.", "danger")
            return render_template("index.html", username=username)

        # Successful login
        session.clear()
        session["user_id"] = user["id"]
        session["username"] = user["username"]
        flash(f"Welcome back, {user['username']}!", "success")
        return redirect(url_for("auth.dashboard"))

    return render_template("index.html")


@auth_bp.route("/logout")
def logout():
    """Log out current user."""
    session.clear()
    flash("You have been logged out successfully.", "info")
    return redirect(url_for("auth.login"))


@auth_bp.route("/dashboard")
@login_required
def dashboard():
    """Protected user dashboard displaying voice enrollment status."""
    user = get_user_by_id(session["user_id"])
    if user is None:
        session.clear()
        return redirect(url_for("auth.login"))

    return render_template("dashboard.html", user=user)

