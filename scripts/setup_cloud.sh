#!/usr/bin/env bash
set -euo pipefail
cd /workspace/project_iis
export UV_CACHE_DIR=/workspace/.cache/uv HF_HOME=/workspace/.cache/huggingface XDG_CACHE_HOME=/workspace/.cache OMP_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false
uv python install 3.11 --install-dir /workspace/toolchains/python --no-bin
if [ ! -x .venv/bin/python ]; then
 /workspace/toolchains/python/cpython-3.11-linux-x86_64-gnu/bin/python3.11 -m venv .venv
fi
uv pip install --python .venv/bin/python torch==2.14.1+cpu --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv/bin/python -r requirements.txt
uv pip check --python .venv/bin/python
.venv/bin/python scripts/prepare_corpus.py
.venv/bin/python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('intfloat/multilingual-e5-small')"
.venv/bin/python scripts/prepare_ollama.py
# Сохраняем существующую конфигурацию; добавляем только отсутствующие настройки.
.venv/bin/python - <<'PY'
from pathlib import Path
from dotenv import dotenv_values, set_key
p=Path('.env')
if not p.exists():p.write_text(Path('.env.example').read_text().replace('LLM_BACKEND=gigachat','LLM_BACKEND=ollama').replace('OLLAMA_MODEL=deepseek-r1:1.5b','OLLAMA_MODEL=iis-bp'))
values=dotenv_values(p)
for k,v in {'HF_HOME':'/workspace/.cache/huggingface','HF_HUB_OFFLINE':'1','OMP_NUM_THREADS':'2','TOKENIZERS_PARALLELISM':'false'}.items():
 if k not in values:set_key(str(p),k,v)
PY
.venv/bin/python ingest.py
.venv/bin/python -m unittest discover -s tests -v
