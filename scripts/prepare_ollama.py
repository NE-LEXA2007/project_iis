"""Установка проверенных CPU-файлов Ollama и GGUF, без отключения TLS."""
import hashlib
import json
import tarfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
META = json.loads((ROOT / 'models.json').read_text())
TARGET = Path('/workspace/toolchains/ollama')
CACHE = Path('/workspace/.cache/ollama')


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        while block := stream.read(1024 * 1024):
            h.update(block)
    return h.hexdigest()


def download(url, path, expected):
    if path.exists() and digest(path) == expected:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + '.part')
    with urllib.request.urlopen(url, timeout=120) as response, partial.open('wb') as output:
        while block := response.read(1024 * 1024):
            output.write(block)
    if digest(partial) != expected:
        partial.unlink()
        raise RuntimeError('SHA-256 не совпадает; артефакт удалён')
    partial.replace(path)


def main():
    files = META['installed_files']
    if not all((TARGET / name).exists() and digest(TARGET / name) == sha for name, sha in files.items()):
        archive = CACHE / 'ollama-linux-amd64.tgz'
        download(META['ollama_url'], archive, META['ollama_sha256'])
        TARGET.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive) as stream:
            selected = []
            for item in stream:
                if Path(item.name).as_posix().removeprefix('./') in files:
                    if not item.isfile() or '..' in Path(item.name).parts or Path(item.name).is_absolute():
                        raise RuntimeError('Небезопасный путь в архиве')
                    selected.append(item)
            stream.extractall(TARGET, members=selected, filter='data')
        archive.unlink()
        if not all((TARGET / name).exists() and digest(TARGET / name) == sha for name, sha in files.items()):
            raise RuntimeError('Проверка установленных файлов не прошла')
    filename = META['file']['rfilename']
    download('https://huggingface.co/' + META['repo'] + '/resolve/' + META['revision'] + '/' + filename,
             CACHE / filename, META['file']['lfs']['sha256'])
    print('Ollama CPU и GGUF: SHA-256 подтверждён')
    print('Запустите ollama serve, затем ollama create iis-bp -f Modelfile')


if __name__ == '__main__':
    main()
