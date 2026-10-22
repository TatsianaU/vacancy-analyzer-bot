"""Контекст диалога по user_id. Хранится в памяти процесса."""

MAX_HISTORY_MESSAGES = 12

_histories: dict[int, list[dict[str, str]]] = {}


def get_history(user_id: int) -> list[dict[str, str]]:
    """Возвращает копию истории пользователя без системного промпта."""
    history = _histories.get(user_id, [])
    return [{"role": item["role"], "content": item["content"]} for item in history]


def add_user_message(user_id: int, text: str) -> None:
    """Добавляет реплику пользователя и обрезает старые сообщения."""
    _append(user_id, "user", text)


def add_assistant_message(user_id: int, text: str) -> None:
    """Добавляет ответ ассистента и обрезает старые сообщения."""
    _append(user_id, "assistant", text)


def clear_history(user_id: int) -> None:
    """Удаляет историю одного пользователя."""
    _histories.pop(user_id, None)


def _append(user_id: int, role: str, text: str) -> None:
    history = _histories.setdefault(user_id, [])
    history.append({"role": role, "content": text})
    _trim(history)


def _trim(history: list[dict[str, str]]) -> None:
    if len(history) > MAX_HISTORY_MESSAGES:
        del history[:-MAX_HISTORY_MESSAGES]
    while history and history[0]["role"] != "user":
        del history[0]
