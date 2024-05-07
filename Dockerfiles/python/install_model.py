from huggingface_hub import snapshot_download, login
import os

REPO_ID = "mistralai/Mistral-7B-v0.1"
TOKEN = os.environ["HUGGINGFACE_TOKEN"]

# Authenticate as the mistral model is restricted
login(TOKEN)
snapshot_download(repo_id=REPO_ID)