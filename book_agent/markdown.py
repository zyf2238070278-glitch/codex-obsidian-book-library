from __future__ import annotations

import re
import unicodedata


_BIDI_CONTROL_CODEPOINTS = frozenset(
    {
        0x061C,
        0x200E,
        0x200F,
        0x202A,
        0x202B,
        0x202C,
        0x202D,
        0x202E,
        0x2066,
        0x2067,
        0x2068,
        0x2069,
    }
)
_MARKDOWN_SYNTAX = frozenset("\\`*_{}[]()!#+-.><|~^=$&")
_URI_SCHEME = re.compile(r"(?i)(?<![\w])(?:[a-z][a-z0-9+.-]{1,31}):(?://)?")


def visible_text(value: object, *, single_line: bool = False) -> str:
    """Make invisible controls visible while retaining the user's readable text."""
    text = str(value).replace("\r\n", "\n").replace("\r", "\n")
    visible: list[str] = []
    for character in text:
        codepoint = ord(character)
        if codepoint in _BIDI_CONTROL_CODEPOINTS:
            visible.append(f"⟦U+{codepoint:04X}⟧")
        elif character == "\t":
            visible.append(" ")
        elif character != "\n" and unicodedata.category(character) == "Cc":
            visible.append(f"⟦U+{codepoint:04X}⟧")
        else:
            visible.append(character)
    rendered = "".join(visible)
    if single_line:
        return " ".join(rendered.splitlines()).strip()
    return rendered


def markdown_literal(value: object, *, single_line: bool = False) -> str:
    """Encode untrusted text so Obsidian/CommonMark renders it as literal prose."""
    rendered = visible_text(value, single_line=single_line)
    uri_delimiters: set[int] = set()
    for match in _URI_SCHEME.finditer(rendered):
        colon = rendered.find(":", match.start(), match.end())
        if colon >= 0:
            uri_delimiters.add(colon)
            if rendered[colon + 1 : colon + 3] == "//":
                uri_delimiters.update((colon + 1, colon + 2))

    escaped: list[str] = []
    for index, character in enumerate(rendered):
        if character in _MARKDOWN_SYNTAX or index in uri_delimiters:
            escaped.append("\\")
        escaped.append(character)
    return "".join(escaped)
