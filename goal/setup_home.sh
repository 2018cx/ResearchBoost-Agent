#!/bin/bash
# Prepare $WORK/<task>/home/.codex for the sandboxed agent: config.toml from codex/config.toml.example and
# codex.env, plus the skill cards named on the command line (default: all of ../skills).
#   usage: setup_home.sh <task> [skill ...]
set -euo pipefail
TASK=$1; shift
HERE=$(cd "$(dirname "$0")" && pwd)
T=${WORK:?set WORK}/$TASK
ENVF=${CODEX_ENV:-$HERE/codex.env}
[ -f "$ENVF" ] || { echo "missing $ENVF: cp $HERE/codex.env.example $HERE/codex.env and fill it in"; exit 1; }
set -a; . "$ENVF"; set +a
: "${OPENAI_API_KEY:?set OPENAI_API_KEY in $ENVF}" "${OPENAI_BASE_URL:?set OPENAI_BASE_URL in $ENVF}"
C=$T/home/.codex; mkdir -p "$C/skills"
python3 - "$HERE/codex/config.toml.example" "$C/config.toml" "$T/ws" <<'PY'
import os, sys
src, dst, ws = sys.argv[1:]
text = open(src).read()
model = os.environ.get('CODEX_MODEL', '').strip()
if model:
    text = text.replace('{CODEX_MODEL}', model)
else:
    text = '\n'.join(l for l in text.split('\n') if '{CODEX_MODEL}' not in l)
text = text.replace('{OPENAI_BASE_URL}', os.environ['OPENAI_BASE_URL']).replace('{WORKSPACE}', ws)
open(dst, 'w').write(text)
PY
SKILLS=$HERE/../skills
for s in ${@:-$(ls "$SKILLS")}; do
  [ -f "$SKILLS/$s/SKILL.md" ] && cp -r "$SKILLS/$s" "$C/skills/"
done
echo "wrote $C/config.toml; skills: $(ls "$C/skills" | tr '\n' ' ')"
