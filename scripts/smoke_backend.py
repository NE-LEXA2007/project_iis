"""Три реплики через MCP, поиск, настоящую модель и память. Это не LLM-судья П6."""
import json
import logging
import re
import sys
import unicodedata
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import config
import memory
from orchestrator import Orchestrator


def check_answer(answer, expected):
    """Ищем целое число, не склеивая цифры разных строк и годов."""
    numbers = re.findall(r'(?<!\w)\d+(?:[ ,\u00a0\u202f]\d{3})*(?!\w)', unicodedata.normalize('NFKC', answer))
    numbers = {re.sub(r'[\s,]', '', value) for value in numbers}
    if expected == 'countries':
        text = answer.casefold()
        return 'росси' in text and 'узбекистан' in text
    return expected in numbers


def main():
    logging.basicConfig(level=logging.INFO)
    o = Orchestrator()
    sid = 'magnit-smoke'
    records = []
    original = o.llm.generate

    def traced(messages, temperature=0.3):
        answer = original(messages, temperature)
        records.append({'messages':messages, 'answer':answer})
        return answer

    o.llm.generate = traced
    cases = [
        ('Какова общая выручка «Магнита» за 2023 год в миллионах рублей?', '2544689'),
        ('Сколько всего магазинов было у группы на конец этого отчётного года?', '29165'),
        ('А в каких странах работала эта сеть магазинов?', 'countries'),
    ]
    try:
        o.reset(sid)
        for question, expected in cases:
            answer = o.handle(sid, question)
            print('BACKEND',config.LLM_BACKEND,'QUESTION',question,'ANSWER',answer,flush=True)
            if not answer or any(marker in answer for marker in ['недоступ', 'не настроен', 'некорректный ответ']):
                raise RuntimeError('Не получен ответ модели')
        assert len(memory.window(sid)) == 6
        assert len(o.usage) == 3
        assert [len(r['messages']) for r in records] == [2, 4, 6]
        content = [check_answer(r['answer'], expected) for r, (_, expected) in zip(records, cases)]
        citations = [bool(re.search(r'\[magnit-[^\]]+\]', r['answer'])) for r in records]
        evidence = {'company':'ПАО «Магнит»','year':2023,'backend':config.LLM_BACKEND,
                    'model':config.OLLAMA_MODEL,'content_checks':content,'citation_checks':citations,
                    'history_sizes':[len(r['messages']) for r in records],'records':records}
        output = ROOT / 'reports' / 'smoke-evidence.json'
        output.write_text(json.dumps(evidence,ensure_ascii=False,indent=2)+'\n')
        print('CONTENT_CHECKS',content,'CITATIONS',citations,flush=True)
        if config.LLM_BACKEND != 'fake':
            assert all(content), 'Ошибка содержания ответа: см. reports/smoke-evidence.json'
            assert all(citations), 'Модель пропустила ссылку: см. reports/smoke-evidence.json'
    finally:
        o.reset(sid)
        o.close()


if __name__ == '__main__':
    main()
