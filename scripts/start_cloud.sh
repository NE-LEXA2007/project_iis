#!/usr/bin/env bash
set -euo pipefail
cd /workspace/project_iis
set -a
source .env.cloud
set +a
mkdir -p /workspace/scratch
if ! curl -fsS http://127.0.0.1:11434/api/version >/dev/null; then
 nohup env OLLAMA_MODELS=/workspace/.cache/ollama/models OLLAMA_HOST=127.0.0.1:11434 OLLAMA_NUM_PARALLEL=1 /workspace/toolchains/ollama/bin/ollama serve > /workspace/scratch/ollama-server.log 2>&1 &
 for attempt in $(seq 1 30); do
  if curl -fsS http://127.0.0.1:11434/api/version >/dev/null; then break; fi
  sleep 1
 done
fi
curl -fsS http://127.0.0.1:11434/api/version
/workspace/toolchains/ollama/bin/ollama create iis-magnit -f Modelfile > /workspace/scratch/ollama-register.log 2>&1
if curl -fsS http://127.0.0.1:8501/_stcore/health >/dev/null; then
 echo "Streamlit уже запущен на порту 8501"
else
 # Передний план сохраняет процесс в долгоживущей сессии облачного терминала.
 exec .venv/bin/python -m streamlit run app.py --server.headless true --server.port 8501 --browser.gatherUsageStats false
fi
