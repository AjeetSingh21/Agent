# Deploying to Streamlit Community Cloud

Community Cloud runs the app straight from the GitHub repo — it reads `requirements.txt`
and `.streamlit/config.toml` on its own, so there is no Dockerfile, no build config, and
no separate deploy branch to maintain.

> **Why not Hugging Face Spaces?** As of 2026 HF deprecated the Streamlit SDK (Streamlit
> apps must now use the Docker SDK), moved Docker and Gradio behind PRO, and removed the
> free CPU Basic tier. Hosting this app there now costs ~$9/month.

## One-time setup

1. Sign in at https://share.streamlit.io with the GitHub account that owns the repo, and
   authorise it.

   The GitHub path is **"Deploy a public app from GitHub"** — make the repo public first.
   Community Cloud no longer offers a free private app; the private option now routes to a
   paid Snowflake trial.

2. **New app** → **Deploy a public app from GitHub**, then:
   - **Repository:** `AjeetSingh21/Agent`
   - **Branch:** `main`
   - **Main file path:** `app.py`
   - **Python version:** 3.11 or later (under *Advanced settings*)
   - **App URL:** pick the subdomain — this becomes the live demo link in the README

3. Still under **Advanced settings**, paste into **Secrets**:

   ```toml
   GROQ_API_KEY = "gsk_your_key_here"
   ```

   Keys must be at the TOML **root level**, not inside a `[section]` — only root-level
   secrets are exported as environment variables, which is how `src/agent/config.py`
   reads them. Add `TAVILY_API_KEY` the same way for better search quality.

   `AGENT_MODEL` does not need to be set; the default in `config.py` is already the
   working model. Override it here only to try a different one.

4. **Deploy**. First build takes 2–5 minutes.

## Every deploy after that

Push to `main`. Community Cloud watches the branch and rebuilds automatically — there is
no deploy script to run.

Changing a secret does **not** rebuild the app. After editing secrets, use
**Manage app → Reboot**.

## Verifying

- Build logs are under **Manage app** (bottom-right of the running app).
- If the sidebar shows "No `GROQ_API_KEY` set", the secret is missing, misspelled, or
  nested under a TOML section. Fix it, then reboot — a secret edit alone won't apply.
- DuckDuckGo rate-limits shared cloud IPs. If searches fail intermittently in the cloud
  but work locally, add a `TAVILY_API_KEY` secret; the agent switches backends
  automatically.

## Known limits of the free tier

- ~1 GB of memory. Fine here — the agent calls Groq over the API and loads no local
  models — but it rules out adding local embeddings or a local model later.
- The app sleeps after 12 hours with no visitors and cold-starts on the next visit
  (~30s). Worth knowing before someone clicks the link from a resume.
- Public apps only. Private hosting is no longer part of the free tier — it redirects to
  a Snowflake trial.
