# Deploy on Streamlit Community Cloud

This copy uses lightweight dependencies (no torch) so it fits the free memory limit.
Embeddings use TF-IDF, so clustering of differently worded answers is less accurate than the full version.

## Steps
1. Revoke any old API keys. Create new ones only for providers you want.
2. Create a GitHub repo and upload the contents of this folder so `app.py` and `requirements.txt` are at the top level.
   Make sure there is no `.env` file in the repo.
3. Go to https://share.streamlit.io and sign in with GitHub.
4. Click **Create app** (or **New app**) and choose your repo, branch `main`, and main file `app.py`.
5. Optional: open **Advanced settings > Secrets** and paste the contents of `.streamlit/secrets.toml.example`,
   replacing the placeholders with your real keys and deleting providers you don't use.
   Keep everything at the top level with straight double quotes.
   Set `KIMI_BASE_URL` to match where your Kimi key was created (`api.moonshot.ai` international, `api.moonshot.cn` China).
   Visitors never enter keys; the app uses only these Secrets. Simulation mode needs no keys.
6. Click **Deploy**. The first build takes a few minutes.

To change secrets later: app menu > **Settings > Secrets**.
