import os
import time
import re
import replicate
import dropbox
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import tempfile

app = Flask(__name__)

# === Dropbox token from Render environment variable ===
DROPBOX_TOKEN = os.getenv("DROPBOX_ACCESS_TOKEN")
REPLICATE_TOKEN = os.getenv("REPLICATE_API_TOKEN")

# === Dropbox Upload ===
def upload_to_dropbox(zip_path, dropbox_path, token):
    dbx = dropbox.Dropbox(token)
    with open(zip_path, "rb") as f:
        dbx.files_upload(f.read(), dropbox_path, mode=dropbox.files.WriteMode.overwrite)
    shared_link_metadata = dbx.sharing_create_shared_link_with_settings(dropbox_path)
    direct_url = re.sub(r"dl=0", "dl=1", shared_link_metadata.url)
    return direct_url

# === Flask route ===
@app.route("/train", methods=["POST"])
def train_from_zip():
    if "file" not in request.files:
        return jsonify({"error": "No ZIP file provided"}), 400

    zip_file = request.files["file"]
    if zip_file.filename == "":
        return jsonify({"error": "Filename is empty"}), 400

    filename = secure_filename(zip_file.filename)
    temp_dir = tempfile.mkdtemp()
    zip_path = os.path.join(temp_dir, filename)
    zip_file.save(zip_path)

    # Upload to Dropbox
    dropbox_dest_path = f"/{filename}"
    try:
        dropbox_url = upload_to_dropbox(zip_path, dropbox_dest_path, DROPBOX_TOKEN)
    except Exception as e:
        return jsonify({"error": f"Dropbox upload failed: {str(e)}"}), 500

    # Start Replicate training
    try:
        client = replicate.Client(api_token=REPLICATE_TOKEN)
        training = client.trainings.create(
            destination="fossilbullet/aifvgusermodel0000001",
            version="ostris/flux-dev-lora-trainer:26dce37af90b9d997eeb970d92e47de3064d46c300504ae376c75bef6a9022d2",
            input={
                "steps": 1000,
                "lora_rank": 16,
                "optimizer": "adamw8bit",
                "batch_size": 1,
                "resolution": "512,768,1024",
                "autocaption": True,
                "input_images": dropbox_url,
                "trigger_word": filename.split(".")[0],  # Make unique per user
                "learning_rate": 0.0004,
                "wandb_project": "flux_train_replicate",
                "wandb_save_interval": 100,
                "caption_dropout_rate": 0.05,
                "cache_latents_to_disk": False,
                "wandb_sample_interval": 100,
                "gradient_checkpointing": False
            }
        )

        return jsonify({
            "status": "training_started",
            "training_url": training.urls.get("get"),
            "training_id": training.id,
            "dropbox_url": dropbox_url
        })

    except Exception as e:
        return jsonify({"error": f"Training failed: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))
