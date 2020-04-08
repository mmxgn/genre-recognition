import os
import time
import classify
import sox
from flask import Flask, request, redirect, jsonify, flash
from werkzeug.utils import secure_filename

# Librosa allowed extensions
ALLOWED_EXTENSIONS = {"wav", "mp3", "ogg"}

app = Flask(__name__)

app.config["UPLOAD_FOLDER"] = "/tmp/uploads"
app.secret_key = "we should probably change this"


def trim_song_in_middle(fname, output_fname, window=5):
    """
        Trims `window' seconds from the middle of the song.
        Supposedly will speed up classification.
    """
    tfm = sox.Transformer()
    duration = sox.file_info.duration(fname)
    tfm.trim(duration / 2 - window / 2, duration / 2 + window / 2)
    tfm.build(fname, output_fname)


@app.route("/", methods=["POST", "GET"])
def index():
    if request.method == "POST":
        started = time.time()
        if "audio" in request.files:

            file = request.files["audio"]

            if file.filename == "":
                flash("No selected file")
                return redirect(request.url)
            if file:

                # In case file is not of the appropriate extension, fail with an error
                if (
                    os.path.splitext(file.filename)[-1][1:].lower()
                    not in ALLOWED_EXTENSIONS
                ):
                    return jsonify(
                        {
                            "status": "failed",
                            "message": "unkown file type: {}".format(
                                os.path.splitext(file.filename)[-1].lower()
                            ),
                            "duration": time.time() - started,
                        }
                    )

                file_path = os.path.join(
                    app.config["UPLOAD_FOLDER"], secure_filename(file.filename)
                )
                # Else, save it to the upload folder, and classify it
                file.save(file_path)
                flash("Classifying: ".format(file_path))
                try:
                    # First trim file
                    path, ext = os.path.splitext(file_path)
                    trimmed_file_path = path + "_trimmed_" + ext

                    trim_song_in_middle(file_path, trimmed_file_path)

                    genre = classify.classify_audio(
                        trimmed_file_path, "models/features_classifier.pkl",
                    )

                    # Return success with the genre of the track
                    reply = {
                        "status": "success",
                        "message": genre,
                        "duration": time.time() - started,
                    }

                except Exception as e:
                    reply = {
                        "status": "failed",
                        "message": "{}".format(e),
                        "duration": time.time() - started,
                    }

                finally:
                    return jsonify(reply)

            return jsonify(
                {
                    "status": "failed",
                    "message": "unspecified error occured",
                    "duration": time.time() - started,
                }
            )

        else:
            return jsonify(
                {
                    "status": "failed",
                    "message": "no files received",
                    "duration": time.time() - started,
                }
            )
