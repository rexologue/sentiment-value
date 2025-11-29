from __future__ import annotations

import re
import html
import unicodedata
from typing import Optional
from dataclasses import dataclass

# --- Опциональные зависимости: ftfy и bs4 ---

try:
    import ftfy  # type: ignore
except ImportError:  # ftfy не установлен
    ftfy = None  # type: ignore

try:
    from bs4 import BeautifulSoup  # type: ignore
    from bs4 import MarkupResemblesLocatorWarning
    import warnings
    warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)
    
except ImportError:  # bs4 не установлен
    BeautifulSoup = None  # type: ignore


# --- Regex'ы для спец-объектов ---

URL_RE = re.compile(
    r"(https?://\S+|www\.\S+)",
    flags=re.IGNORECASE,
)

USER_RE = re.compile(
    r"@\w+",
    flags=re.UNICODE,
)

EMAIL_RE = re.compile(
    r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+",
    flags=re.UNICODE,
)

# Русский номер: +7 (904) 535 35 35, +7 904 535 35 35, 8 800 535 35 35 и подобные
RU_PHONE_RE = re.compile(
    r"""
    (?:
        (?:\+7|8)              # код страны/транк: +7 или 8
        [\s\-()]*
        \d{3}                  # код оператора / региона: 3 цифры
        (?:[\s\-()]*
            \d{3}              # следующая тройка
        )
        (?:[\s\-()]*
            \d{2}              # пара цифр
        )
        (?:[\s\-()]*
            \d{2}              # ещё пара
        )
    )
    """,
    flags=re.VERBOSE,
)

# Более общий fallback-паттерн: 6+ цифр с пробелами/дефисами/скобками
GENERIC_PHONE_RE = re.compile(
    r"""
    (?:
        (?:\+?\d{1,3})?        # опциональный код страны
        [\s\-()]*
    )?
    (?:\d[\s\-()]*){6,}        # 6+ цифр с пробелами/дефисами/скобками
    """,
    flags=re.VERBOSE,
)


@dataclass
class NormalizationConfig:
    # --- базовые вещи ---
    unicode_normalize: bool = True
    strip_control_chars: bool = True

    # HTML
    strip_html_tags: bool = True       # убирать HTML-теги
    html_unescape: bool = True         # &amp; -> &, &quot; -> "
    use_bs4_html_parser: bool = True   # если bs4 доступен — использовать его

    # ftfy: починить битый текст
    use_ftfy: bool = True

    # знаки препинания и буквы
    normalize_quotes: bool = True
    normalize_dashes: bool = True
    yo_to_e: bool = True

    # токены-заглушки
    replace_urls: bool = True
    replace_users: bool = True
    replace_emails: bool = True
    replace_phones: bool = True

    url_token: str = "<URL>"
    user_token: str = "<USER>"
    email_token: str = "<EMAIL>"
    phone_token: str = "<NUM>"

    # пробелы
    normalize_whitespace: bool = True


def _strip_control_chars(text: str) -> str:
    """Убрать не-печатные Unicode-символы (категория C), но оставить \n и \t."""
    return "".join(
        ch
        for ch in text
        if not (
            unicodedata.category(ch)[0] == "C"
            and ch not in ("\n", "\t")
        )
    )


def _strip_html_tags_regex(text: str) -> str:
    """Простой regex-удалитель тегов <...>. Используется, если bs4 недоступен."""
    return re.sub(r"<[^>\n]+>", " ", text)


def _strip_html_tags_bs4(text: str) -> str:
    """Удаление HTML через BeautifulSoup, если доступен."""
    if BeautifulSoup is None:
        # fallback
        return _strip_html_tags_regex(text)
    # parser можно поменять на "html.parser", если не хочешь тащить lxml
    soup = BeautifulSoup(text, "lxml")
    return soup.get_text(" ")


def _normalize_punctuation(text: str, cfg: NormalizationConfig) -> str:
    if cfg.normalize_quotes:
        # разные кавычки -> " / '
        quotes_map = {
            "«": '"',
            "»": '"',
            "„": '"',
            "“": '"',
            "”": '"',
            "‟": '"',
            "‘": "'",
            "’": "'",
        }
        for src, dst in quotes_map.items():
            text = text.replace(src, dst)

    if cfg.normalize_dashes:
        # разные тире/минусы -> обычный дефис
        dashes = ["–", "—", "−", "‒", "―"]
        for d in dashes:
            text = text.replace(d, "-")

    if cfg.yo_to_e:
        text = text.replace("ё", "е").replace("Ё", "Е")

    return text


def _replace_special_tokens(text: str, cfg: NormalizationConfig) -> str:
    if cfg.replace_urls:
        text = URL_RE.sub(cfg.url_token, text)

    if cfg.replace_users:
        text = USER_RE.sub(cfg.user_token, text)

    if cfg.replace_emails:
        text = EMAIL_RE.sub(cfg.email_token, text)

    if cfg.replace_phones:
        # Сначала более специфичный русский формат,
        # потом общий fallback на "много цифр".
        text = RU_PHONE_RE.sub(cfg.phone_token, text)
        text = GENERIC_PHONE_RE.sub(cfg.phone_token, text)

    return text


def _normalize_whitespace(text: str) -> str:
    # \s включает \n, поэтому после этого все переносы превратятся в пробелы.
    # Если нужно сохранять \n — можно сделать отдельную функцию/флаг.
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_text(text: str, cfg: Optional[NormalizationConfig] = None) -> str:
    """
    Основная функция нормализации под Sentiment Analysis / BERT-подобные модели.

    Делает:
    - ftfy.fix_text (если доступен и включён)
    - html.unescape (&amp; -> & и т.п.)
    - Unicode NFKC
    - удаление управляющих символов
    - снятие HTML-тегов (через bs4, если есть, иначе regex)
    - замену URL/@user/email/phone на токены
    - нормализацию кавычек, тире, ё->е
    - нормализацию пробелов
    """
    if cfg is None:
        cfg = NormalizationConfig()

    if not isinstance(text, str):
        text = str(text)

    # 0) Починка битого текста через ftfy (если доступен)
    if cfg.use_ftfy and ftfy is not None:
        text = ftfy.fix_text(text)

    # 1) HTML entities: &amp; -> &
    if cfg.html_unescape:
        text = html.unescape(text)

    # 2) Unicode-нормализация
    if cfg.unicode_normalize:
        text = unicodedata.normalize("NFKC", text)

    # 3) Удаление управляющих символов
    if cfg.strip_control_chars:
        text = _strip_control_chars(text)

    # 4) Снятие HTML-тегов
    if cfg.strip_html_tags:
        if cfg.use_bs4_html_parser and BeautifulSoup is not None:
            text = _strip_html_tags_bs4(text)
        else:
            text = _strip_html_tags_regex(text)

    # 5) Замена спец-объектов (URL, @user, email, phone)
    text = _replace_special_tokens(text, cfg)

    # 6) Нормализация кавычек/тире/ё
    text = _normalize_punctuation(text, cfg)

    # 7) Нормализация пробелов
    if cfg.normalize_whitespace:
        text = _normalize_whitespace(text)

    return text
