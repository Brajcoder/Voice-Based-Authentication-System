"""
Flask entry point for Voice-Based Authentication System.
Integrates SQLite, Auth routes, Voice ML routes, and template rendering.
"""

import os
from flask import Flask, redirect, url_for, session, render_template, jsonify
from database import init_db
from extensions import limiter
from auth import auth_bp
from voice import voice_bp


def create_app(test_config=None) -> Flask:
    """Application factory for Voice Authentication System."""
    secret_key = None
    if test_config and "SECRET_KEY" in test_config:
        secret_key = test_config["SECRET_KEY"]
    else:
        secret_key = os.environ.get("SECRET_KEY")

    if not secret_key:
        raise RuntimeError(
            "SECRET_KEY environment variable is not set. "
            "Please set the SECRET_KEY environment variable before running the application."
        )

    app = Flask(__name__, instance_relative_config=True)

    # Configuration
    app.config.from_mapping(
        SECRET_KEY=secret_key,
        MAX_CONTENT_LENGTH=16 * 1024 * 1024,  # Max 16MB upload
    )

    if test_config:
        app.config.from_mapping(test_config)

    # Initialize Database and Extensions
    init_db(app)
    limiter.init_app(app)

    # Register Blueprints
    app.register_blueprint(auth_bp)
    app.register_blueprint(voice_bp)

    @app.route("/")
    def index():
        """Root route: redirect to dashboard if logged in, else login page."""
        if "user_id" in session:
            return redirect(url_for("auth.dashboard"))
        return redirect(url_for("auth.login"))

    @app.errorhandler(429)
    def ratelimit_handler(e):
        """Custom 429 rate limit exceeded handler returning standardized JSON."""
        return jsonify({
            "success": False,
            "error": "Too many attempts. Try again later."
        }), 429

    @app.errorhandler(404)
    def page_not_found(e):
        return render_template("index.html", error="Page not found."), 404

    @app.errorhandler(500)
    def internal_error(e):
        return render_template("index.html", error="Internal server error occurred."), 500

    return app


try:
    app = create_app()
except RuntimeError:
    app = None

if __name__ == "__main__":
    if app is None:
        app = create_app()
    print("=" * 60)
    print(" Starting Voice-Based Authentication System")
    print(" Server running on http://127.0.0.1:5000")
    print("=" * 60)
    app.run(debug=True, host="127.0.0.1", port=5000)

