"""Telegram-бот: команды, проверка текста и выдача разбора вакансии."""

import asyncio
import logging
import sys

from aiogram import Bot, Dispatcher, F
from aiogram.exceptions import TelegramAPIError
from aiogram.filters import Command
from aiogram.types import Message
from openai import AsyncOpenAI

from api_client import AnalysisError, request_model_response
from config import ConfigError, Settings, load_settings
from context_manager import (
    add_assistant_message,
    add_user_message,
    clear_history,
    get_history,
)

MAX_USER_MESSAGE_LENGTH = 12_000
TELEGRAM_CHUNK_SIZE = 4_000

START_TEXT = (
    "Я разбираю текст вакансии или заказа на работу.\n\n"
    "Пришлите объявление одним сообщением. В ответе будут суть, задачи, "
    "функции, нужные знания, инструменты, явные требования, предположения, "
    "возможные этапы, пробелы и вопросы заказчику.\n\n"
    "Факты из текста, предположения и то, что нужно уточнить, я разделяю. "
    "Того, чего в объявлении нет, я не додумываю как факт.\n\n"
    "После разбора можно задавать вопросы по этой же вакансии. "
    "Перед новым объявлением отправьте /reset.\n"
    "Инструкция: /help"
)

HELP_TEXT = (
    "Как пользоваться:\n"
    "1. Скопируйте полный текст вакансии или заказа.\n"
    "2. Отправьте его обычным сообщением.\n"
    "3. Дождитесь разбора.\n"
    "4. Задавайте уточняющие вопросы в этом же диалоге.\n"
    "5. Перед новым объявлением отправьте /reset.\n\n"
    "Команды:\n"
    "/start — что умеет бот\n"
    "/help — эта инструкция\n"
    "/reset — стереть контекст диалога\n\n"
    f"Одно сообщение — не длиннее {MAX_USER_MESSAGE_LENGTH} символов. "
    "Бот смотрит только присланный текст и не ищет информацию в интернете."
)

RESET_TEXT = "Контекст этого диалога очищен. Пришлите следующее объявление."
EMPTY_TEXT = "Пустое сообщение не о чем разбирать. Пришлите текст вакансии или вопрос по ней."
UNKNOWN_COMMAND_TEXT = "Такой команды нет. Доступны /start, /help и /reset."
NON_TEXT = "Нужен текст объявления или вопрос к нему. Пришлите обычное текстовое сообщение."
PROCESSING_NEW = "Разбираю объявление. Это займёт несколько секунд."
PROCESSING_FOLLOWUP = "Отвечаю на вопрос с учётом предыдущего разбора."
GENERIC_ERROR_TEXT = "Не получилось обработать сообщение. Попробуйте ещё раз."

logger = logging.getLogger(__name__)
dp = Dispatcher()
_locks: dict[int, asyncio.Lock] = {}


@dp.message(Command("start"))
async def handle_start(message: Message) -> None:
    await message.answer(START_TEXT)


@dp.message(Command("help"))
async def handle_help(message: Message) -> None:
    await message.answer(HELP_TEXT)


@dp.message(Command("reset"))
async def handle_reset(message: Message) -> None:
    user_id = _user_id(message)
    if user_id is None:
        return
    async with _user_lock(user_id):
        clear_history(user_id)
    logger.info("user_id=%s context_cleared=true", user_id)
    await message.answer(RESET_TEXT)


@dp.message(F.text)
async def handle_text(
    message: Message,
    settings: Settings,
    openai_client: AsyncOpenAI,
) -> None:
    user_id = _user_id(message)
    if user_id is None:
        return

    text = (message.text or "").strip()
    if not text:
        await message.answer(EMPTY_TEXT)
        return
    if text.startswith("/"):
        await message.answer(UNKNOWN_COMMAND_TEXT)
        return
    if len(text) > MAX_USER_MESSAGE_LENGTH:
        await message.answer(
            f"Сообщение слишком длинное: {len(text)} символов. "
            f"Лимит — {MAX_USER_MESSAGE_LENGTH}. Сократите текст и отправьте его снова."
        )
        return

    async with _user_lock(user_id):
        status = await message.answer(
            PROCESSING_FOLLOWUP if get_history(user_id) else PROCESSING_NEW
        )
        history = [*get_history(user_id), {"role": "user", "content": text}]
        try:
            result = await request_model_response(
                history,
                settings=settings,
                client=openai_client,
                user_id=user_id,
            )
            add_user_message(user_id, text)
            add_assistant_message(user_id, result.text)
            await _deliver(message, status, result.text)
        except AnalysisError as error:
            await _replace_status(status, message, error.user_message)
        except Exception as error:
            logger.error(
                "user_id=%s handler_error error_type=%s",
                user_id,
                type(error).__name__,
            )
            await _replace_status(status, message, GENERIC_ERROR_TEXT)


@dp.message()
async def handle_non_text(message: Message) -> None:
    await message.answer(NON_TEXT)


def split_text(text: str, limit: int = TELEGRAM_CHUNK_SIZE) -> list[str]:
    """Делит длинный ответ на части, приемлемые для Telegram."""
    remaining = text.strip()
    if not remaining:
        return []
    if limit < 1:
        raise ValueError("limit must be positive")

    chunks: list[str] = []
    while remaining:
        if len(remaining) <= limit:
            chunks.append(remaining)
            break
        window = remaining[:limit]
        split_at = _split_index(window, limit)
        chunk = remaining[:split_at].strip()
        if not chunk:
            chunk = remaining[:limit]
            split_at = limit
        chunks.append(chunk)
        remaining = remaining[split_at:].strip()
    return chunks


def _split_index(window: str, limit: int) -> int:
    for separator in ("\n\n", "\n", " "):
        position = window.rfind(separator)
        if position >= limit // 2:
            return position
    return limit


async def _deliver(message: Message, status: Message, text: str) -> None:
    parts = split_text(text) or [GENERIC_ERROR_TEXT]
    await _replace_status(status, message, parts[0])
    for part in parts[1:]:
        await message.answer(part)


async def _replace_status(status: Message, message: Message, text: str) -> None:
    try:
        await status.edit_text(text)
    except TelegramAPIError:
        await message.answer(text)


def _user_id(message: Message) -> int | None:
    if message.from_user is None:
        return None
    return message.from_user.id


def _user_lock(user_id: int) -> asyncio.Lock:
    lock = _locks.get(user_id)
    if lock is None:
        lock = asyncio.Lock()
        _locks[user_id] = lock
    return lock


def _configure_stdio() -> None:
    """Позволяет показать русскую ошибку конфигурации в консоли Windows."""
    if sys.platform == "win32":
        try:
            import ctypes

            ctypes.windll.kernel32.SetConsoleOutputCP(65001)
            ctypes.windll.kernel32.SetConsoleCP(65001)
        except (AttributeError, OSError):
            pass
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (OSError, ValueError):
            pass


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    # Подробный журнал HTTP-клиентов может содержать адрес запроса.
    for logger_name in ("httpx", "httpcore", "httpx2", "httpcore2", "openai"):
        logging.getLogger(logger_name).setLevel(logging.WARNING)


async def run_bot(settings: Settings) -> None:
    bot = Bot(token=settings.bot_token)
    async with AsyncOpenAI(
        api_key=settings.openai_api_key,
        timeout=90.0,
        max_retries=0,
    ) as client:
        await dp.start_polling(bot, settings=settings, openai_client=client)


def main() -> None:
    _configure_stdio()
    _configure_logging()
    try:
        settings = load_settings()
    except ConfigError as error:
        print(error, file=sys.stderr)
        raise SystemExit(1) from None

    logger.info(
        "Bot started. model=%s temperature=%s max_tokens=%s",
        settings.openai_model,
        settings.temperature,
        settings.max_tokens,
    )
    asyncio.run(run_bot(settings))


if __name__ == "__main__":
    main()
