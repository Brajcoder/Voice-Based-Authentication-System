"""
Voice enrollment and verification routes for Voice-Based Authentication System.
Handles audio ingestion, ML embedding generation, cosine similarity scoring,
and speaker verification thresholds.
"""

from flask import Blueprint, request, jsonify, session
from database import (
    get_user_by_id,
    get_user_by_username,
    save_voice_embedding,
    delete_voice_embedding,
)
from auth import login_required
from ml.preprocess import preprocess_audio
from ml.model import (
    extract_embedding,
    compute_similarity,
    embedding_to_bytes,
    bytes_to_embedding,
)

voice_bp = Blueprint("voice", __name__, url_prefix="/voice")

# Voice matching threshold for cosine similarity with Resemblyzer d-vectors
SIMILARITY_THRESHOLD = 0.6993

def _get_audio_from_request():
    """Extract audio data from either multipart form upload or raw body."""
    if "audio" in request.files:
        audio_file = request.files["audio"]
        if audio_file.filename == "":
            raise ValueError("No audio file selected.")
        return audio_file.read()

    # Raw audio bytes in request body
    data = request.get_data()
    if data and len(data) > 0:
        return data

    raise ValueError("No audio payload received in request.")


@voice_bp.route("/enroll", methods=["POST"])
@login_required
def enroll():
    """
    Enroll the logged-in user's voice signature.
    Receives an audio sample, extracts the CNN embedding, and stores in SQLite.
    """
    user_id = session.get("user_id")
    user = get_user_by_id(user_id)
    if not user:
        return jsonify({"success": False, "error": "User not found."}), 404

    try:
        audio_data = _get_audio_from_request()

        # Preprocess and clean audio
        cleaned_signal, sr = preprocess_audio(audio_data)

        # Extract 128-d normalized embedding
        embedding = extract_embedding(cleaned_signal, sr=sr)

        # Serialize and save to database
        raw_bytes = embedding_to_bytes(embedding)
        success = save_voice_embedding(user_id, raw_bytes)

        if not success:
            return jsonify({"success": False, "error": "Database update failed."}), 500

        return jsonify({
            "success": True,
            "message": "Voice signature successfully enrolled!",
            "is_enrolled": 1,
        })

    except ValueError as e:
        return jsonify({"success": False, "error": str(e)}), 400
    except Exception as e:
        return jsonify({"success": False, "error": f"Voice processing error: {str(e)}"}), 500


@voice_bp.route("/verify", methods=["POST"])
def verify():
    """
    Verify voice audio against stored enrollment.
    Supports two modes:
    1. Authenticated user testing voice match from dashboard.
    2. Unauthenticated user logging in via voice ("Login with Voice") with username.
    """
    username = request.form.get("username", "").strip()
    is_login_attempt = False

    if "user_id" in session:
        # Dashboard test mode
        user = get_user_by_id(session["user_id"])
    elif username:
        # Voice login mode
        user = get_user_by_username(username)
        is_login_attempt = True
    else:
        return jsonify({
            "success": False,
            "error": "Please provide a username or log in first.",
        }), 400

    if not user:
        return jsonify({
            "success": False,
            "error": "User not found.",
        }), 404

    if not user["is_enrolled"] or not user["voice_embedding"]:
        return jsonify({
            "success": False,
            "error": "No voice signature enrolled for this user. Please enroll first.",
        }), 400

    try:
        audio_data = _get_audio_from_request()

        # Preprocess incoming test audio
        cleaned_signal, sr = preprocess_audio(audio_data)

        # Extract embedding from test audio
        test_embedding = extract_embedding(cleaned_signal, sr=sr)

        # Retrieve enrolled reference embedding
        enrolled_embedding = bytes_to_embedding(user["voice_embedding"])

        # Compute cosine similarity
        score = compute_similarity(enrolled_embedding, test_embedding)

        # Map score [-1, 1] to human-friendly percentage [0, 100]
        # For speech similarity, score >= 0.72 indicates high confidence match
        percentage = max(0.0, min(100.0, float((score + 1.0) / 2.0 * 100)))

        is_match = bool(score >= SIMILARITY_THRESHOLD)

        # If voice login was requested and matched, establish session
        if is_login_attempt and is_match:
            session.clear()
            session["user_id"] = user["id"]
            session["username"] = user["username"]

        return jsonify({
            "success": True,
            "matched": is_match,
            "score": round(float(score), 4),
            "percentage": round(percentage, 1),
            "threshold": SIMILARITY_THRESHOLD,
            "message": (
                "Voice verified successfully! Match confirmed."
                if is_match
                else "Voice verification failed. Speaker does not match enrolled profile."
            ),
            "logged_in": is_login_attempt and is_match,
            "redirect_url": "/dashboard" if (is_login_attempt and is_match) else None,
        })

    except ValueError as e:
        return jsonify({"success": False, "error": str(e)}), 400
    except Exception as e:
        return jsonify({"success": False, "error": f"Verification error: {str(e)}"}), 500


@voice_bp.route("/reset", methods=["POST"])
@login_required
def reset():
    """Reset and delete current user's enrolled voice signature."""
    user_id = session.get("user_id")
    success = delete_voice_embedding(user_id)
    if success:
        return jsonify({"success": True, "message": "Voice profile removed. You can now re-enroll."})
    return jsonify({"success": False, "error": "Failed to reset voice profile."}), 500


@voice_bp.route("/status", methods=["GET"])
def status():
    """Get enrollment status for current user or specified username."""
    if "user_id" in session:
        user = get_user_by_id(session["user_id"])
    else:
        username = request.args.get("username", "").strip()
        user = get_user_by_username(username) if username else None

    if not user:
        return jsonify({"enrolled": False, "user_found": False})

    return jsonify({
        "user_found": True,
        "username": user["username"],
        "enrolled": bool(user["is_enrolled"]),
        "enrolled_at": str(user["enrolled_at"]) if user["enrolled_at"] else None,
    })
