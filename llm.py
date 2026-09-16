"""Обработчик вызова DeepSeek: валидация ввода, запрос к модели, карта ошибок."""

from __future__ import annotations

import logging
import os
import ssl
from pathlib import Path

import httpx
from dotenv import load_dotenv
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    OpenAI,
)

load_dotenv(Path(__file__).resolve().parent / ".env")

logger = logging.getLogger(__name__)

MAX_DISH_LENGTH = 500
FORMAT_LIST = "Список продуктов"
FORMAT_STEPS = "Список + шаги"

MSG_EMPTY = "Напишите, что хотите приготовить."
MSG_TOO_LONG = "Слишком длинный текст — сократите или разбейте на части."
MSG_ACCESS = "Не настроен доступ к сервису."
MSG_UNAVAILABLE = "Сервис временно недоступен. Попробуйте позже."
MSG_NETWORK = "Не удалось получить ответ. Попробуйте ещё раз."
MSG_EMPTY_RESPONSE = "Пустой ответ модели. Попробуйте переформулировать вопрос."

_SYSTEM_PROMPT = (
    "Вы помогаете составить список покупок по блюду. "
    "Отвечайте только текстом результата, без предисловий и без markdown. "
    "Каждый продукт — отдельная строка: название и количество с учётом числа порций."
)


class AppError(Exception):
    """Ошибка с коротким сообщением для пользователя (без traceback в UI)."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class _RetryServer(Exception):
    """Внутренний сигнал: 5xx, можно повторить запрос один раз."""


class _RetryNetwork(Exception):
    """Внутренний сигнал: таймаут или сеть, можно повторить запрос один раз."""


def generate_shopping_list(dish: str, people: str, format: str) -> str:
    """Вернуть список покупок (и шаги, если выбран соответствующий формат)."""
    cleaned = (dish or "").strip()
    if not cleaned:
        raise AppError(MSG_EMPTY)
    if len(cleaned) > MAX_DISH_LENGTH:
        raise AppError(MSG_TOO_LONG)

    provider = (os.getenv("LLM_PROVIDER") or "deepseek").strip().lower()
    api_key = (os.getenv("LLM_API_KEY") or "").strip()
    if provider != "deepseek" or not api_key:
        raise AppError(MSG_ACCESS)

    user_prompt = _build_user_prompt(cleaned, people, format)
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    last_retry: type[Exception] | None = None
    for attempt in range(2):
        try:
            return _call_deepseek(api_key, messages)
        except _RetryServer:
            last_retry = _RetryServer
            if attempt == 0:
                continue
        except _RetryNetwork:
            last_retry = _RetryNetwork
            if attempt == 0:
                continue

    if last_retry is _RetryServer:
        raise AppError(MSG_UNAVAILABLE)
    raise AppError(MSG_NETWORK)


def _build_user_prompt(dish: str, people: str, format: str) -> str:
    parts = [
        f"Блюдо: {dish}",
        f"На сколько человек: {people}",
        f"Формат ответа: {format}",
        "Требование к списку продуктов: по одной строке на продукт, количества под указанное число порций.",
    ]
    if format == FORMAT_STEPS:
        parts.append(
            "После списка продуктов добавьте пустую строку, заголовок "
            "«Шаги приготовления» и 5–8 коротких шагов (по одному на строку)."
        )
    else:
        parts.append("Только список продуктов, без шагов приготовления.")
    return "\n".join(parts)


def _call_deepseek(api_key: str, messages: list[dict[str, str]]) -> str:
    try:
        timeout = float(os.getenv("LLM_TIMEOUT", "30"))
        model = (os.getenv("LLM_MODEL") or "deepseek-flash").strip()
        base_url = (os.getenv("LLM_BASE_URL") or "https://api.deepseek.com").strip()
        client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=0,
        )
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            stream=False,
            extra_body={"thinking": {"type": "disabled"}},
        )
        return _extract_text(response)
    except AppError:
        raise
    except APIStatusError as exc:
        status = getattr(exc, "status_code", None)
        if isinstance(status, int) and status >= 500:
            logger.warning("DeepSeek HTTP %s", status)
            raise _RetryServer() from None
        logger.warning("DeepSeek HTTP client error")
        raise AppError(MSG_ACCESS) from None
    except (APITimeoutError, httpx.TimeoutException, TimeoutError):
        logger.warning("DeepSeek timeout")
        raise _RetryNetwork() from None
    except (APIConnectionError, httpx.NetworkError, httpx.TransportError, ConnectionError):
        logger.warning("DeepSeek network error")
        raise _RetryNetwork() from None
    except ssl.SSLError:
        logger.warning("DeepSeek SSL error")
        raise AppError(MSG_ACCESS) from None


def _extract_text(response: object) -> str:
    try:
        choices = getattr(response, "choices", None)
        if choices:
            message = getattr(choices[0], "message", None)
            text = _content_to_text(getattr(message, "content", None) if message else None)
            if text:
                return text
    except (IndexError, AttributeError, TypeError):
        pass
    raise AppError(MSG_EMPTY_RESPONSE)


def _content_to_text(content: object) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        chunks: list[str] = []
        for part in content:
            if isinstance(part, str):
                chunks.append(part)
                continue
            text = getattr(part, "text", None)
            if text is None and isinstance(part, dict):
                text = part.get("text")
            if isinstance(text, str):
                chunks.append(text)
        return "".join(chunks).strip()
    return ""
