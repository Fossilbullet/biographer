import os
import time
import re
import replicate
import dropbox
import requests
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import tempfile
import zipfile

app = Flask(__name__)

# === Dropbox and Replicate tokens from environment ===
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

# === Download and Zip Images ===
def download_and_zip_images(image_urls, temp_dir):
    zip_path = os.path.join(temp_dir, "images.zip")
    with zipfile.ZipFile(zip_path, "w") as zipf:
        for i, url in enumerate(image_urls):
            try:
                response = requests.get(url)
                if response.status_code == 200:
                    ext = url.split(".")[-1].split("?")[0]
                    img_filename = f"image_{i}.{ext}"
                    img_path = os.path.join(temp_dir, img_filename)
                    with open(img_path, "wb") as f:
                        f.write(response.content)
                    zipf.write(img_path, arcname=img_filename)
            except Exception as e:
                print(f"Failed to download {url}: {e}")
    return zip_path

# === Flask route ===
@app.route("/train", methods=["POST"])
def train_from_urls():
    data = request.get_json()

    if not data or "image_urls" not in data:
        return jsonify({"error": "No image URLs provided"}), 400

    image_urls = data["image_urls"]
    if not isinstance(image_urls, list) or not all(isinstance(url, str) for url in image_urls):
        return jsonify({"error": "image_urls must be a list of strings"}), 400

    temp_dir = tempfile.mkdtemp()

    try:
        # Download and zip images
        zip_path = download_and_zip_images(image_urls, temp_dir)
        filename = f"user_upload_{int(time.time())}.zip"

        # Upload to Dropbox
        dropbox_dest_path = f"/{filename}"
        dropbox_url = upload_to_dropbox(zip_path, dropbox_dest_path, DROPBOX_TOKEN)

        # Start Replicate training
        client = replicate.Client(api_token=REPLICATE_TOKEN)
        trigger_word = filename.split(".")[0]

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
                "trigger_word": trigger_word,
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
