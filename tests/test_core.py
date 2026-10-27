"""Проверки реализации без сети; финансовые строки здесь синтетические."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, Mock

import config
import memory
import prompts
from ingest import clean_text, chunk_text, detect_category
from contracts import Chunk, Message
from llm_client import OllamaClient, GigaChatClient
from orchestrator import Orchestrator


class Documents(unittest.TestCase):
    def test_clean_preserves_units_and_headings(self):
        raw = '# Примечание 4 / 2023 / млн руб.\n\nПоказатель | 2023 | 2022\nВыручка | 100 | 90\n\nДолгосроч-\nные обязательства\n23\n'
        text = clean_text(raw)
        self.assertIn('Долгосрочные обязательства', text)
        self.assertNotIn('\n23', text)
        self.assertIn('Показатель | 2023 | 2022\nВыручка', text)
        self.assertGreaterEqual(text.count('Примечание 4'), 2)

    def test_windows_cover_long_text(self):
        text = ''.join(chr(0x400 + i) for i in range(200))
        chunks = chunk_text(text, size=50, overlap=10)
        self.assertEqual(chunks[0][-10:], chunks[1][:10])
        self.assertEqual(chunks[-1][-1], text[-1])
        self.assertTrue(all(len(c) <= 50 for c in chunks))
        for size, overlap in [(0, 0), (10, 10), (10, -1)]:
            with self.assertRaises(ValueError):
                chunk_text('abc', size, overlap)
        self.assertEqual(chunk_text(''), [])

    def test_units_not_split_unnecessarily(self):
        units = ['а' * 25, 'б' * 25, 'в' * 25]
        self.assertEqual(chunk_text('\n\n'.join(units), 55, 0), [units[0] + '\n\n' + units[1], units[2]])

    def test_categories(self):
        for category in ['financials', 'segments', 'risks', 'events']:
            self.assertEqual(detect_category('выручка', category + '.md'), category)
        self.assertEqual(detect_category('неизвестный текст', 'other.txt'), 'general')


class Routing(unittest.TestCase):
    def setUp(self):
        self.o = object.__new__(Orchestrator)
        self.o._state = {}

    def test_domain_phrases(self):
        cases = [('Какова выручка?', 'question'), ('Структура долга', 'question'),
                 ('Выручка сегмента', 'question'), ('Ключевые риски', 'risk'),
                 ('Какие угрозы названы?', 'risk'), ('Что с дивидендами?', 'event'),
                 ('Существенные факты года', 'event'), ('Общее собрание', 'event'), ('Какие события произошли за год?', 'event'),
                 ('Сравни цифры', 'question'), ('Привет', 'question')]
        for text, expected in cases:
            with self.subTest(text=text):
                self.assertEqual(self.o.detect_intent(text), expected)

    def test_forbidden_transition_and_session_isolation(self):
        self.assertEqual(self.o.route('a', 'risk'), 'risk_analyst')
        with self.assertLogs('orchestrator', level='WARNING') as logs:
            self.assertEqual(self.o.route('a', 'event'), 'analyst')
        self.assertIn('откат', '\n'.join(logs.output))
        self.assertEqual(self.o.route('b', 'event'), 'events_analyst')
        self.assertEqual(self.o.route('a', 'unknown'), 'analyst')


class Memory(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.patch = patch.object(config, 'DB_PATH', Path(self.tmp.name) / 'memory.db')
        self.patch.start()
        memory.create_schema()

    def tearDown(self):
        self.patch.stop()
        self.tmp.cleanup()

    def test_persistence_in_new_process(self):
        import subprocess
        import sys
        memory.save('persist', 'user', 'сохранилось')
        script = "import config; from pathlib import Path; config.DB_PATH=Path(" + repr(str(config.DB_PATH)) + "); import memory; assert memory.window('persist')[0].content == 'сохранилось'"
        subprocess.run([sys.executable, '-c', script], check=True)

    def test_order_limit_sessions_and_clear(self):
        for i in range(10):
            memory.save('one', 'user', str(i))
        memory.save('two', 'user', "O'Reilly; DROP TABLE messages;")
        self.assertEqual([m.content for m in memory.window('one', 4)], ['6', '7', '8', '9'])
        self.assertEqual(len(memory.window('one', 0)), 0)
        self.assertEqual(len(memory.window('two')), 1)
        memory.clear('one')
        self.assertEqual(memory.window('one'), [])
        self.assertEqual(len(memory.window('two')), 1)


class PromptsAndClients(unittest.TestCase):
    def test_messages_and_empty_context(self):
        for role in Orchestrator.ROLES:
            messages = prompts.build_messages(role, 'Вопрос?', [Chunk('Текст', 'doc.md', 0, 'general')],
                                              [Message('user', 'Раньше'), Message('assistant', 'Ответ')])
            self.assertEqual([m['role'] for m in messages], ['system', 'user', 'assistant', 'user'])
            self.assertTrue(messages[-1]['content'].endswith('Вопрос: Вопрос?'))
            self.assertNotIn('стр. 0', messages[-1]['content'])
        self.assertIn('ничего не вернула', prompts.build_messages('analyst', '?', [], [])[-1]['content'])

    @patch('llm_client.requests.post')
    def test_ollama_payload_and_think(self, post):
        post.return_value.json.return_value = {'message': {'content': '<think>analysis</think>Ответ'}}
        self.assertEqual(OllamaClient().generate([{'role': 'user', 'content': '?'}], 0.1), 'Ответ')
        self.assertFalse(post.call_args.kwargs['json']['stream'])
        self.assertEqual(post.call_args.kwargs['json']['options']['temperature'], 0.1)
        post.return_value.json.return_value = {}
        self.assertIn('некорректный', OllamaClient().generate([]))

    @patch('gigachat.GigaChat')
    def test_gigachat_payload(self, sdk):
        sdk.return_value.chat.return_value.choices = [Mock(message=Mock(content='Ответ'))]
        sdk.return_value.tokens_count.return_value = [Mock(tokens=12)]
        with patch.object(config, 'GIGACHAT_CREDENTIALS', 'unit-test-only'):
            client = GigaChatClient()
            self.assertEqual(client.generate([], 0), 'Ответ')
            self.assertEqual(sdk.return_value.chat.call_args.args[0]['temperature'], 0.01)
            self.assertEqual(client.count_tokens('текст'), 12)
        self.assertTrue(sdk.call_args.kwargs['verify_ssl_certs'])


class RussianCorpus(unittest.TestCase):
    def test_number_check_does_not_concatenate_different_values(self):
        from scripts.smoke_backend import check_answer
        self.assertTrue(check_answer('Выручка 2 544 689 млн рублей', '2544689'))
        self.assertTrue(check_answer('Выручка 2,544,689 млн рублей', '2544689'))
        self.assertFalse(check_answer('2 магазина, 544 сотрудника, 689 дней', '2544689'))
        self.assertFalse(check_answer('2023 и 2022 годы', '20232022'))
        self.assertTrue(check_answer('Россия и Республика Узбекистан', 'countries'))
        self.assertFalse(check_answer('Только Россия', 'countries'))

    def test_active_corpus_has_no_bp_files_and_has_local_sources(self):
        import json
        from ingest import read_txt
        root = Path(__file__).resolve().parents[1]
        manifest = json.loads((root / 'sources.json').read_text())
        self.assertEqual(manifest['company'], 'ПАО «Магнит»')
        self.assertTrue(all(name.startswith('magnit-') for name in manifest['raw_files']))
        for name in manifest['transcripts']:
            text = (root / 'corpus' / name).read_text()
            self.assertIn('Учебный перевод', text)
            self.assertIn('Источник:', text)
        cases = json.loads((root / 'gold_dataset.json').read_text())
        self.assertEqual(len(cases), 5)
        for case in cases:
            self.assertTrue(set(case['expected_sources']) <= set(manifest['raw_files']))
        self.assertEqual(len(cases[2]['expected_sources']), 2)


if __name__ == '__main__':
    unittest.main()
