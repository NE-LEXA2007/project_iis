"""Восстановить учебную выборку, проверяя TLS и SHA-256 исходного PDF."""
import hashlib
import json
import shutil
import urllib.request
from pathlib import Path
from pypdf import PdfReader, PdfWriter

ROOT = Path(__file__).resolve().parents[1]


def main():
    manifest = json.loads((ROOT / 'sources.json').read_text())
    cache = ROOT / 'data' / 'downloads'
    cache.mkdir(parents=True, exist_ok=True)
    original = cache / 'bp-annual-2023.pdf'
    if not original.exists():
        request = urllib.request.Request(manifest['url'], headers={'User-Agent': 'IIS-course-project/1.0'})
        with urllib.request.urlopen(request, timeout=90) as response:
            content = response.read()
        if hashlib.sha256(content).hexdigest() != manifest['sha256']:
            raise SystemExit('SHA-256 не совпадает: исходник не сохранён, требуется проверка версии источника')
        original.write_bytes(content)
    if hashlib.sha256(original.read_bytes()).hexdigest() != manifest['sha256']:
        raise SystemExit('SHA-256 сохранённого PDF не совпадает')
    reader = PdfReader(original)
    raw = ROOT / 'data' / 'raw'
    raw.mkdir(parents=True, exist_ok=True)
    for name, pages in manifest['extracts'].items():
        writer = PdfWriter()
        for page in pages:
            writer.add_page(reader.pages[page - 1])
        with (raw / name).open('wb') as stream:
            writer.write(stream)
        print(name, len(pages), 'страниц')
    for path in (ROOT / 'corpus').glob('*.md'):
        shutil.copyfile(path, raw / path.name)
        print(path.name, 'транскрипция таблицы')


if __name__ == '__main__':
    main()
