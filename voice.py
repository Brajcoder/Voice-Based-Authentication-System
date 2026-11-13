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
    embeddings_list_to_bytes,
    bytes_to_embeddings_list,
    increment_voice_fail,
    reset_voice_fail,
    is_voice_locked,
)
from auth import login_required
from extensions import limiter
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
@limiter.limit("10 per minute")
@login_required
def enroll():
    """
    Enroll the logged-in user's voice signature.
    Stores up to 3 voice embeddings per user.
    If fewer than 3 samples exist, appends the new embedding to the list.
    If 3 already exist, replaces the oldest (FIFO).
    Maintains backward compatibility with the single voice_embedding column.
    """
    user_id = session.get("user_id")
    user = get_user_by_id(user_id)
    if not user:
        return jsonify({"success": False, "error": "User not found."}), 404

    try:
        audio_data = _get_audio_from_request()

        # Preprocess and clean audio
        cleaned_signal, sr = preprocess_audio(audio_data)

        # Extract normalized embedding
        new_embedding = extract_embedding(cleaned_signal, sr=sr)

        # Retrieve existing embeddings from list column
        existing_embeddings = []
        if "voice_embeddings_list" in user.keys() and user["voice_embeddings_list"]:
            existing_embeddings = bytes_to_embeddings_list(user["voice_embeddings_list"])

        # Backward compatibility: fall back to single voice_embedding if list is empty
        if not existing_embeddings and "voice_embedding" in user.keys() and user["voice_embedding"]:
            try:
                single_emb = bytes_to_embedding(user["voice_embedding"])
                existing_embeddings = [single_emb]
            except Exception:
                existing_embeddings = []

        # If fewer than 3 samples exist append, if 3 already exist replace oldest
        if len(existing_embeddings) < 3:
            existing_embeddings.append(new_embedding)
        else:
            existing_embeddings.pop(0)
            existing_embeddings.append(new_embedding)

        # Serialize list of numpy arrays for voice_embeddings_list column
        list_bytes = embeddings_list_to_bytes(existing_embeddings)
        # Also serialize latest single embedding for backward compatibility with voice_embedding column
        single_bytes = embedding_to_bytes(new_embedding)

        # Save to database
        success = save_voice_embedding(
            user_id,
            embedding_bytes=single_bytes,
            embeddings_list_bytes=list_bytes,
        )

        if not success:
            return jsonify({"success": False, "error": "Database update failed."}), 500

        count = len(existing_embeddings)
        return jsonify({
            "success": True,
            "message": f"Voice signature sample {count}/3 successfully enrolled!",
            "is_enrolled": 1,
            "samples_count": count,
        })

    except ValueError as e:
        return jsonify({"success": False, "error": str(e)}), 400
    except Exception as e:
        return jsonify({"success": False, "error": f"Voice processing error: {str(e)}"}), 500


@voice_bp.route("/verify", methods=["POST"])
@limiter.limit("5 per minute")
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

    # Before processing audio check if the user is locked
    lock_status = is_voice_locked(user["id"])
    if lock_status:
        mins = lock_status.remaining_minutes
        minute_str = f"{mins} minute{'s' if mins != 1 else ''}"
        return jsonify({
            "success": False,
            "error": f"Voice login is locked due to too many failed attempts. Try again in {minute_str}.",
            "locked": True,
            "remaining_minutes": mins,
        }), 403

    has_embeddings = bool(
        ("voice_embeddings_list" in user.keys() and user["voice_embeddings_list"])
        or ("voice_embedding" in user.keys() and user["voice_embedding"])
    )
    if not user["is_enrolled"] or not has_embeddings:
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

        # Retrieve enrolled reference embeddings
        stored_embeddings = []
        if "voice_embeddings_list" in user.keys() and user["voice_embeddings_list"]:
            stored_embeddings = bytes_to_embeddings_list(user["voice_embeddings_list"])

        # Backward compatibility with existing single voice_embedding BLOB column
        if not stored_embeddings and "voice_embedding" in user.keys() and user["voice_embedding"]:
            try:
                stored_embeddings = [bytes_to_embedding(user["voice_embedding"])]
            except Exception:
                stored_embeddings = []

        if not stored_embeddings:
            return jsonify({
                "success": False,
                "error": "No valid voice embeddings found for user. Please enroll first.",
            }), 400

        # During verification in compute_similarity, compare the test embedding
        # against all stored embeddings and use the maximum cosine similarity score as the final score.
        score = compute_similarity(stored_embeddings, test_embedding)

        # Map score [-1, 1] to human-friendly percentage [0, 100]
        # For speech similarity, score >= 0.72 indicates high confidence match
        percentage = max(0.0, min(100.0, float((score + 1.0) / 2.0 * 100)))

        is_match = bool(score >= SIMILARITY_THRESHOLD)

        if is_match:
            # On a successful match reset the counter
            reset_voice_fail(user["id"])
            if is_login_attempt:
                session.clear()
                session["user_id"] = user["id"]
                session["username"] = user["username"]
            msg = "Voice verified successfully! Match confirmed."
            is_locked = False
            remaining_mins = 0
        else:
            # On a failed match increment the counter
            fail_result = increment_voice_fail(user["id"])
            is_locked = fail_result.locked
            remaining_mins = fail_result.remaining_minutes
            if is_locked:
                minute_str = f"{remaining_mins} minute{'s' if remaining_mins != 1 else ''}"
                msg = f"Voice verification failed. Maximum attempts exceeded. Voice login is locked for {minute_str}."
            else:
                attempts_left = max(0, 5 - fail_result.count)
                msg = f"Voice verification failed. Speaker does not match enrolled profile. ({attempts_left} attempt{'s' if attempts_left != 1 else ''} remaining before lock)"

        return jsonify({
            "success": True,
            "matched": is_match,
            "score": round(float(score), 4),
            "percentage": round(percentage, 1),
            "threshold": SIMILARITY_THRESHOLD,
            "message": msg,
            "locked": is_locked,
            "remaining_minutes": remaining_mins,
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

    has_embeddings = bool(
        ("voice_embeddings_list" in user.keys() and user["voice_embeddings_list"])
        or ("voice_embedding" in user.keys() and user["voice_embedding"])
    )
    enrolled = bool(user["is_enrolled"] and has_embeddings)
    samples_count = 0
    if "voice_embeddings_list" in user.keys() and user["voice_embeddings_list"]:
        samples_count = len(bytes_to_embeddings_list(user["voice_embeddings_list"]))
    elif "voice_embedding" in user.keys() and user["voice_embedding"]:
        samples_count = 1

    lock_status = is_voice_locked(user["id"])
    return jsonify({
        "user_found": True,
        "username": user["username"],
        "enrolled": enrolled,
        "samples_count": samples_count,
        "enrolled_at": str(user["enrolled_at"]) if user["enrolled_at"] else None,
        "locked": bool(lock_status),
        "remaining_minutes": lock_status.remaining_minutes if lock_status else 0,
    })
