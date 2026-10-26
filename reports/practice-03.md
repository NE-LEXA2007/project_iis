# Практика 3 — проверка автомата

| Фраза | Интент | Роль из начального состояния |
| --- | --- | --- |
| Какова выручка BP? | question | analyst |
| Структура долга | question | analyst |
| Выручка сегмента | question | analyst |
| Ключевые риски | risk | risk_analyst |
| Какие угрозы названы? | risk | risk_analyst |
| Что с дивидендами? | event | events_analyst |
| Существенные факты года | event | events_analyst |
| Общее собрание | event | events_analyst |
| Сравни цифры | question | analyst |
| Привет | question | analyst |

Запрещённый переход: risk_analyst → events_analyst откатывается в analyst. В журнале получено предупреждение «Переход risk_analyst -> events_analyst запрещён, откат в analyst». Риск сам по себе не доказывает свершившееся событие; нужна сверка финансовым аналитиком.

Поиск каждой роли: analyst — весь индекс; risk_analyst — category=risks; events_analyst — category=events. Переходы определены графом, detect_intent не вызывает LLM.
