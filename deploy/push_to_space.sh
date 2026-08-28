#!/usr/bin/env bash
# Push the current branch to the Hugging Face Space with the HF-flavoured README.
# Leaves the working tree and the GitHub README untouched.
set -euo pipefail

cd "$(dirname "$0")/.."

if ! git remote get-url space >/dev/null 2>&1; then
  echo "No 'space' remote configured. See deploy/DEPLOY.md step 3." >&2
  exit 1
fi

if [ -n "$(git status --porcelain)" ]; then
  echo "Working tree is dirty. Commit or stash first." >&2
  exit 1
fi

branch=$(git rev-parse --abbrev-ref HEAD)
tmp="space-deploy-$$"

git checkout -q -b "$tmp"
cp deploy/HF_SPACE_README.md README.md
git add README.md
git commit -q -m "Deploy to Hugging Face Space"
git push -f space "$tmp:main"

git checkout -q "$branch"
git branch -q -D "$tmp"
git checkout -q -- README.md

echo "Pushed to Space. Build progress: Space -> Logs tab."
