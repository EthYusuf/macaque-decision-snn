"""Arayüz dili (tr / en): konsol mesajları, eğitim kayıtları ve şekiller.

CLI'de `--lang en` veya ortam değişkeni MONKEYBRAIN_LANG=en ile seçilir.
"""

import os

LANG = os.environ.get("MONKEYBRAIN_LANG", "tr")


def set_lang(lang: str) -> None:
    global LANG
    LANG = lang


def pick(tr: str, en: str) -> str:
    """Seçili dildeki metni döndürür."""
    return en if LANG == "en" else tr
