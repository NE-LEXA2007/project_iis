"""Три прогона с изменением одного параметра; финальная конфигурация восстанавливается."""
import functools
import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import config
import ingest


def main():
    original_chunk = ingest.chunk_text
    original_size, original_overlap = config.CHUNK_SIZE, config.CHUNK_OVERLAP
    # Чтение PDF не зависит от параметров нарезки.
    ingest.READERS['.pdf'] = functools.lru_cache(None)(ingest.read_pdf)
    results = []
    try:
        for size, overlap in [(512, 64), (768, 64), (768, 128)]:
            config.CHUNK_SIZE, config.CHUNK_OVERLAP = size, overlap
            ingest.chunk_text = functools.partial(original_chunk, size=size, overlap=overlap)
            ingest.ingest()
            collection = config.get_collection()
            rate5 = ingest.evaluate_retrieval(5)
            rate1 = ingest.evaluate_retrieval(1)
            filtered = {}
            for category in ['financials', 'segments', 'risks', 'events']:
                response = collection.query(query_texts=['Магнит 2023'], n_results=5, where={'category': category})
                metas = response['metadatas'][0]
                assert metas and all(m['category'] == category for m in metas), category
                filtered[category] = len(metas)
            cross = json.loads((ROOT / 'gold_dataset.json').read_text())[2]
            response = collection.query(query_texts=[cross['question']], n_results=5)
            found = {m['source'] for m in response['metadatas'][0]}
            results.append({'size':size,'overlap':overlap,'chunks':collection.count(),
                            'hit5':rate5,'hit1':rate1,'filters':filtered,
                            'cross_all_sources_at5':set(cross['expected_sources']).issubset(found),
                            'cross_found':sorted(found)})
            (ROOT / 'reports' / 'retrieval-results.json').write_text(json.dumps(results, ensure_ascii=False, indent=2)+'\n')
    finally:
        config.CHUNK_SIZE, config.CHUNK_OVERLAP = original_size, original_overlap
        ingest.chunk_text = original_chunk
    # Возвращаем индекс в состояние, объявленное в config.py.
    ingest.ingest()
    ingest.evaluate_retrieval()


if __name__ == '__main__':
    main()
