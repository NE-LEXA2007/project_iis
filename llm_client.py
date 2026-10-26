"""
llm_client.py — доступ к языковой модели за общим интерфейсом.

ПРАКТИКА 4. Что делает студент:
  1. GigaChatClient.generate() — вызов Сбер GigaChat через официальный SDK
  2. OllamaClient.generate()   — вызов локальной модели через HTTP API Ollama
  3. count_tokens()            — оценка длины промпта до отправки

Готово и трогать не нужно: build_client() и выбор бэкенда по LLM_BACKEND.

Критерии приёмки:
  - смена LLM_BACKEND с gigachat на ollama не требует правки orchestrator.py
  - длина промпта считается до отправки и логируется
  - оба клиента принимают один и тот же список messages и возвращают строку

НЕ МЕНЯТЬ: сигнатуру generate(messages, temperature) — она объявлена в contracts.py
и на неё опирается оркестратор.

Зачем абстракция. Учебные квоты кончаются, сеть в аудитории падает, а занятие
должно продолжаться. Ollama — фолбэк, который работает офлайн. Заодно это
материал для сравнения авторегрессионной модели и reasoning-модели из Л4:
у DeepSeek-R1 в ответе виден блок рассуждений, у GigaChat — нет.

Третий бэкенд — fake_llm.FakeClient, офлайн-заглушка без сети и ключей. Она
нужна для отладки, когда недоступны оба клиента, и ваши реализации не заменяет:
работа, сданная на LLM_BACKEND=fake, не принимается.
"""

import logging
import re
import requests

import config

log = logging.getLogger(__name__)
from contracts import LLMClient


class GigaChatClient:
    """Клиент Сбер GigaChat API."""

    def __init__(self):
        self.model = config.GIGACHAT_MODEL
        from gigachat import GigaChat
        self._client = GigaChat(credentials=config.GIGACHAT_CREDENTIALS,
                                model=self.model,
                                verify_ssl_certs=config.GIGACHAT_VERIFY_SSL,
                                timeout=60)

    def generate(self, messages: list[dict], temperature: float = 0.3) -> str:
        """Отправить диалог в GigaChat и вернуть текст ответа.

        SDK принимает структуру вида {"messages": [...], "temperature": ...}.
        Ответ лежит в response.choices[0].message.content.
        Не забудьте обработать сетевую ошибку: занятие не должно падать трассировкой.
        """
        if not config.GIGACHAT_CREDENTIALS:
            return "GigaChat не настроен: задайте GIGACHAT_CREDENTIALS в локальном .env."
        try:
            response = self._client.chat({"messages": messages,
                                          "temperature": max(0.01, temperature)})
            return response.choices[0].message.content
        except Exception as exc:
            # Не выводим payload, URL с параметрами или текст SDK-ошибки:
            # они могут содержать пользовательские данные или авторизацию.
            log.warning("Ошибка GigaChat: %s", type(exc).__name__)
            return "GigaChat временно недоступен. Проверьте подключение и настройки доступа."

    def count_tokens(self, text: str) -> int:
        """Оценка длины в токенах.

        У GigaChat есть метод tokens_count. Если он недоступен, дайте оценку:
        для русского текста примерно один токен на 3 символа. Точность здесь
        не главное — важно видеть порядок величины и связать его со стоимостью.
        """
        if config.GIGACHAT_CREDENTIALS:
            try:
                return int(self._client.tokens_count([text], model=self.model)[0].tokens)
            except Exception as exc:
                log.warning("Оценка токенов GigaChat: %s", type(exc).__name__)
        return max(1, (len(text) + 2) // 3)


class OllamaClient:
    """Клиент локального инференса через Ollama."""

    def __init__(self):
        self.model = config.OLLAMA_MODEL
        self.host = config.OLLAMA_HOST

    def generate(self, messages: list[dict], temperature: float = 0.3) -> str:
        """POST на {host}/api/chat с телом {"model", "messages", "stream": False}.

        Ответ — в data["message"]["content"].
        У reasoning-моделей ответ приходит с блоком <think>...</think> — решите,
        показывать его пользователю или вырезать, и обоснуйте выбор на защите.
        """
        try:
            response = requests.post(self.host.rstrip("/") + "/api/chat",
                                     json={"model": self.model, "messages": messages,
                                           "stream": False,
                                           "options": {"temperature": temperature}},
                                     timeout=(10, 180))
            response.raise_for_status()
            content = response.json()["message"]["content"]
            # Внутреннее рассуждение не является проверенным ответом.
            content = re.sub(r"<think>.*?</think>", "", content, flags=re.S)
            if "<think>" in content:
                return "Модель не завершила ответ. Повторите запрос."
            return content.strip() or "Модель вернула пустой ответ. Повторите запрос."
        except (requests.RequestException, ValueError, KeyError, TypeError) as exc:
            log.warning("Ошибка Ollama: %s", type(exc).__name__)
            return "Ollama недоступна или вернула некорректный ответ. Проверьте сервер и модель."

    def count_tokens(self, text: str) -> int:
        """Оценка длины в токенах — та же логика, что и в GigaChatClient."""
        return max(1, (len(text) + 2) // 3)


def build_client() -> LLMClient:
    """Выбрать бэкенд по конфигурации. Готово, менять не нужно."""
    if config.LLM_BACKEND == "ollama":
        return OllamaClient()
    if config.LLM_BACKEND == "gigachat":
        return GigaChatClient()
    if config.LLM_BACKEND == "fake":
        # Импорт внутри ветки: без него fake_llm тянулся бы при любом бэкенде
        from fake_llm import FakeClient

        return FakeClient()
    raise ValueError(
        f"Неизвестный LLM_BACKEND: {config.LLM_BACKEND!r}. "
        f"Допустимо: gigachat | ollama | fake"
    )
