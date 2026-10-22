"""Загрузка настроек бота из окружения и файла .env."""

import math
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# Единственное место в коде, где задано имя модели по умолчанию.
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
DEFAULT_TEMPERATURE = 0.3
DEFAULT_MAX_TOKENS = 1200
MAX_ALLOWED_TOKENS = 128_000

_ENV_PATH = Path(__file__).resolve().parent / ".env"


class ConfigError(Exception):
    """Понятная ошибка конфигурации. Не содержит ключей и токенов."""


@dataclass(frozen=True)
class Settings:
    """Настройки процесса. Секреты скрыты в текстовом представлении."""

    bot_token: str
    openai_api_key: str
    openai_model: str
    temperature: float
    max_tokens: int

    def __repr__(self) -> str:
        return (
            "Settings(bot_token='***', openai_api_key='***', "
            f"openai_model={self.openai_model!r}, temperature={self.temperature!r}, "
            f"max_tokens={self.max_tokens!r})"
        )


def load_settings() -> Settings:
    """Читает .env и проверяет обязательные переменные."""
    load_dotenv(_ENV_PATH)

    problems: list[str] = []
    bot_token = os.getenv("BOT_TOKEN", "").strip()
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not bot_token:
        problems.append("не задан BOT_TOKEN")
    if not api_key:
        problems.append("не задан OPENAI_API_KEY")

    model = os.getenv("OPENAI_MODEL", "").strip() or DEFAULT_OPENAI_MODEL
    temperature, temperature_problem = _parse_temperature(os.getenv("TEMPERATURE"))
    max_tokens, max_tokens_problem = _parse_max_tokens(os.getenv("MAX_TOKENS"))
    if temperature_problem:
        problems.append(temperature_problem)
    if max_tokens_problem:
        problems.append(max_tokens_problem)

    if problems:
        raise ConfigError(
            "Бот не запущен. Проверьте файл .env: "
            + "; ".join(problems)
            + ". Скопируйте .env.example в .env и заполните обязательные "
            "BOT_TOKEN и OPENAI_API_KEY. Сами ключи в это сообщение не выводятся."
        )

    return Settings(
        bot_token=bot_token,
        openai_api_key=api_key,
        openai_model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def _parse_temperature(raw: str | None) -> tuple[float, str | None]:
    if raw is None or not raw.strip():
        return DEFAULT_TEMPERATURE, None
    normalized = raw.strip().replace(",", ".")
    try:
        value = float(normalized)
    except ValueError:
        return DEFAULT_TEMPERATURE, "TEMPERATURE должна быть числом от 0 до 2, например 0.3"
    if not math.isfinite(value) or not 0 <= value <= 2:
        return DEFAULT_TEMPERATURE, "TEMPERATURE должна быть числом от 0 до 2, например 0.3"
    return value, None


def _parse_max_tokens(raw: str | None) -> tuple[int, str | None]:
    if raw is None or not raw.strip():
        return DEFAULT_MAX_TOKENS, None
    try:
        value = int(raw.strip())
    except ValueError:
        return (
            DEFAULT_MAX_TOKENS,
            "MAX_TOKENS должен быть целым числом больше нуля, например 1200",
        )
    if not 1 <= value <= MAX_ALLOWED_TOKENS:
        return (
            DEFAULT_MAX_TOKENS,
            f"MAX_TOKENS должен быть целым числом от 1 до {MAX_ALLOWED_TOKENS}",
        )
    return value, None
