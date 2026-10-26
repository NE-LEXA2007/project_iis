"""Сквозной вызов MCP → ChromaDB → LLM; явно указывает выбранный бэкенд."""
import logging
import json
import re
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
import memory
from orchestrator import Orchestrator

logging.basicConfig(level=logging.INFO)
o = Orchestrator()
sid = 'smoke-readiness'
records = []
original_generate = o.llm.generate

def traced_generate(messages, temperature=0.3):
    # Контекст для проверки памяти; финансовые вопросы тестовые, секретов нет.
    records.append({'messages': messages})
    answer = original_generate(messages, temperature)
    records[-1]['answer'] = answer
    return answer

o.llm.generate = traced_generate
try:
    o.reset(sid)
    for question in ['Каков Net debt BP на конец 2023 года в миллионах долларов?',
                     'А Net debt на конец 2022 года в тех же единицах?',
                     'Сравни значения Net debt из двух предыдущих ответов: как изменилась сумма в миллионах долларов?']:
        answer = o.handle(sid, question)
        print('BACKEND',config.LLM_BACKEND,'QUESTION',question,'ANSWER',answer,flush=True)
        if not answer or any(marker in answer for marker in ['недоступ', 'не настроен', 'некорректный ответ']):
            raise RuntimeError('Не получен ответ модели')
    # Проверяем переданный в модель диалог, отдельно от качества генерации.
    assert len(memory.window(sid)) == 6
    assert len(o.usage) == 3
    assert [len(r['messages']) for r in records] == [2, 4, 6]
    if config.LLM_BACKEND != 'fake':
        expected = [('20912',), ('21422',), ('20912', '21422')]
        quality = []
        for record, values in zip(records, expected):
            digits = re.sub(r'\D', '', record['answer'])
            quality.append(all(value in digits for value in values))
        arithmetic_ok = '510' in re.sub(r'\D', '', records[-1]['answer'])
        print('NUMERIC_CHECKS', quality, 'ARITHMETIC_CHECK', arithmetic_ok, flush=True)
        Path('/workspace/scratch/smoke-evidence.json').write_text(
            json.dumps({'backend':config.LLM_BACKEND,'model':config.OLLAMA_MODEL,
                        'numeric_checks':quality,'arithmetic_check':arithmetic_ok,'records':records},ensure_ascii=False,indent=2))
        assert all(quality) and arithmetic_ok, 'Ошибка чисел или расчёта модели: смотрите smoke-evidence.json' 
finally:
    o.reset(sid)
    o.close()
