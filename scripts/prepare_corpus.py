"""Восстановить только выбранный русский корпус; TLS и SHA-256 обязательны."""
import hashlib
import json
import shutil
import urllib.request
from pathlib import Path
from pypdf import PdfReader, PdfWriter

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def prepare(root=ROOT):
    manifest = json.loads((root / 'sources.json').read_text(encoding='utf-8'))
    if manifest.get('schema_version') != 2:
        raise ValueError('Нужен sources.json версии 2')
    cache = root / 'data' / 'downloads'
    cache.mkdir(parents=True, exist_ok=True)
    raw = root / 'data' / 'raw'
    raw.mkdir(parents=True, exist_ok=True)
    for document in manifest['documents']:
        original = cache / document['filename']
        if original.name != document['filename']:
            raise ValueError('filename должен быть именем файла без каталогов')
        if not original.exists():
            request = urllib.request.Request(document['url'], headers={'User-Agent': 'IIS-course-project/1.0'})
            with urllib.request.urlopen(request, timeout=90) as response:
                content = response.read()
            if hashlib.sha256(content).hexdigest() != document['sha256']:
                raise ValueError('Изменилась версия источника: SHA-256 не совпадает')
            original.write_bytes(content)
        if sha256(original) != document['sha256']:
            raise ValueError('SHA-256 сохранённого источника не совпадает')
        if document.get('extracts'):
            reader = PdfReader(original)
            for name, pages in document['extracts'].items():
                if Path(name).name != name:
                    raise ValueError('Недопустимое имя выборки')
                writer = PdfWriter()
                for page in pages:
                    writer.add_page(reader.pages[page - 1])
                with (raw / name).open('wb') as stream:
                    writer.write(stream)
                print(name, len(pages), 'страниц')
    for name in manifest['transcripts']:
        if Path(name).name != name:
            raise ValueError('Недопустимое имя транскрипции')
        shutil.copyfile(root / 'corpus' / name, raw / name)
        print(name, 'проверенная выдержка')
    for name in manifest['raw_files']:
        if not (raw / name).is_file():
            raise FileNotFoundError(raw / name)
    return manifest


if __name__ == '__main__':
    prepare()
