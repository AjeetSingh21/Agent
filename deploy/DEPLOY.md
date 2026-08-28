# Deploying to Hugging Face Spaces

The Space needs its own `README.md` carrying YAML frontmatter that HF reads as
configuration. That frontmatter is kept out of the GitHub README (where it would render
as an odd table) and lives in `deploy/HF_SPACE_README.md` instead.

## One-time setup

1. Create the Space at https://huggingface.co/new-space
   - **SDK:** Streamlit
   - **Hardware:** CPU basic (free)
   - **Visibility:** Public

2. Add your Groq key as a secret:
   Space → **Settings** → **Variables and secrets** → **New secret**
   - Name: `GROQ_API_KEY`
   - Value: your key from https://console.groq.com/keys

   Add `TAVILY_API_KEY` the same way if you want better search quality.

3. Add the Space as a git remote and push:

```bash
git remote add space https://huggingface.co/spaces/YOUR_USERNAME/multi-step-research-agent
```

## Every deploy

The Space README must be swapped in before pushing. Run this from the repo root:

```bash
bash deploy/push_to_space.sh
```

It builds a detached commit with the HF README in place, pushes it to the Space, and
leaves your working tree and GitHub README untouched.

## Verifying

- The Space build log is under the **Logs** tab; a cold build takes 2-4 minutes.
- If the app loads but the sidebar shows "No GROQ_API_KEY set", the secret name is wrong
  or was added after the last build — restart the Space from **Settings → Factory reboot**.
- DuckDuckGo occasionally rate-limits shared cloud IPs. If searches fail intermittently
  on the Space but work locally, add a `TAVILY_API_KEY` secret; the agent switches
  backends automatically.
