from __future__ import annotations

import json
import os
import re
import secrets
import stat
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from book_agent.config import AppPaths
from book_agent.storage import Database
from book_agent.vault import _managed_directory_beneath


_TITLE_CATEGORY_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("色彩科学与调色", ("调色", "colour book", "colour sense", "color correction")),
    ("虚拟现实与数字媒体", ("虚拟现实", "vr影像", "vr 影像")),
    ("摄影艺术与史论", ("摄影史", "摄影师", "照片的本质", "photograph")),
    ("电视与视频工程", ("电视原理", "电视技术", "广播电视")),
    ("影视制作与技术", ("影视技术", "视频技术", "电影制作")),
    ("艺术理论", ("艺术学", "艺术理论", "introduction to art")),
    ("小说", ("小说", "旧日之主")),
)

_PREVIEW_CATEGORY_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("色彩科学与调色", ("色彩校正", "colour sense", "color grading")),
    ("虚拟现实与数字媒体", ("虚拟现实", "virtual reality")),
    ("摄影艺术与史论", ("摄影艺术", "摄影史", "photographic")),
    ("电视与视频工程", ("电视原理", "广播电视工程")),
    ("影视制作与技术", ("影视制作", "电影制作技术", "视频技术")),
    ("艺术理论", ("艺术学概论", "艺术理论")),
    ("小说", ("旧日支配者", "网络小说")),
)


def classify_book(title: str, author: str | None, preview: str) -> str:
    """Return one deterministic primary category from bounded book metadata."""

    title_haystack = "\n".join((title, author or "")).casefold()
    for category, terms in _TITLE_CATEGORY_RULES:
        if any(term.casefold() in title_haystack for term in terms):
            return category
    preview_haystack = preview[:4000].casefold()
    for category, terms in _PREVIEW_CATEGORY_RULES:
        if any(term.casefold() in preview_haystack for term in terms):
            return category
    return "待分类"


_BOOK_ID = re.compile(r"[0-9a-f]{24}")
_UNSAFE_FILENAME = re.compile(r'[\\/:*?"<>|\x00-\x1f]')
_MAX_FILENAME_BYTES = 255


@dataclass(frozen=True)
class CatalogSyncResult:
    total: int
    created: int
    updated: int
    failed: int
    errors: tuple[str, ...] = ()


def _yaml_string(value: object) -> str:
    return json.dumps(str(value), ensure_ascii=False)


def _frontmatter(text: str) -> list[str]:
    lines = text.splitlines()
    if not lines or lines[0] != "---":
        raise ValueError("existing catalog card must start with YAML frontmatter")
    try:
        end = lines.index("---", 1)
    except ValueError as exc:
        raise ValueError("existing catalog card has unterminated YAML frontmatter") from exc
    return lines[1:end]


def _parse_scalar(value: str, name: str) -> str:
    stripped = value.strip()
    if not stripped:
        raise ValueError(f"{name} must not be blank")
    if stripped.startswith('"'):
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{name} is not a valid string") from exc
        if type(parsed) is not str or not parsed.strip():
            raise ValueError(f"{name} must not be blank")
        return parsed
    return stripped


def _user_categories(text: str) -> tuple[str, tuple[str, ...]]:
    lines = _frontmatter(text)
    primary: str | None = None
    custom: tuple[str, ...] | None = None
    index = 0
    while index < len(lines):
        line = lines[index]
        if line.startswith("primary_category:"):
            primary = _parse_scalar(line.partition(":")[2], "primary_category")
        elif line.startswith("custom_categories:"):
            value = line.partition(":")[2].strip()
            if value:
                try:
                    parsed = json.loads(value)
                except json.JSONDecodeError as exc:
                    raise ValueError("custom_categories must be a list") from exc
                if type(parsed) is not list or any(
                    type(item) is not str or not item.strip() for item in parsed
                ):
                    raise ValueError("custom_categories must contain nonblank strings")
                custom = tuple(parsed)
            else:
                items: list[str] = []
                cursor = index + 1
                while cursor < len(lines) and lines[cursor].startswith("  - "):
                    items.append(
                        _parse_scalar(lines[cursor][4:], "custom_categories item")
                    )
                    cursor += 1
                custom = tuple(items)
                index = cursor - 1
        index += 1
    if primary is None:
        raise ValueError("existing catalog card is missing primary_category")
    if custom is None:
        raise ValueError("existing catalog card is missing custom_categories")
    return primary, custom


class CatalogService:
    def __init__(
        self,
        paths: AppPaths,
        database: Database,
        *,
        vault_root_identity: tuple[int, int] | None = None,
    ) -> None:
        self.paths = paths
        self.database = database
        self._vault_root_identity = vault_root_identity

    def _validate_vault_root(self) -> None:
        try:
            info = self.paths.vault.lstat()
        except OSError as exc:
            raise ValueError("Managed vault root is unavailable") from exc
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
            raise ValueError("Managed vault root must be a real directory")
        identity = (info.st_dev, info.st_ino)
        if self._vault_root_identity is None:
            self._vault_root_identity = identity
        elif identity != self._vault_root_identity:
            raise ValueError(f"Managed vault root identity changed: {self.paths.vault}")

    def sync_book(self, book: Mapping[str, Any], preview: str = "") -> Path:
        self._validate_vault_root()
        book_id = str(book.get("book_id") or "")
        if _BOOK_ID.fullmatch(book_id) is None:
            raise ValueError("book_id must be exactly 24 lowercase hexadecimal characters")
        title = str(book.get("title") or "").strip()
        if not title:
            raise ValueError("title must not be blank")
        with self._catalog_directory(create=True) as (directory, directory_fd):
            card_name = self._card_name(title, book_id, directory_fd)
            existing = self._read_card(directory_fd, card_name)
            if existing is not None:
                primary, custom = _user_categories(existing)
            else:
                author_value = book.get("author")
                primary = classify_book(
                    title,
                    None if author_value is None else str(author_value),
                    preview,
                )
                custom = ()
            content = self._render(book, primary=primary, custom=custom)
            self._write_atomically(directory_fd, card_name, content)
            return directory / card_name

    def sync_all(self) -> CatalogSyncResult:
        self._validate_vault_root()
        books = self.database.list_books()
        created = 0
        updated = 0
        errors: list[str] = []
        for book in books:
            book_id = str(book.get("book_id") or "")
            try:
                existed = self._card_exists(book_id)
                self.sync_book(book, preview=self._preview(book.get("parsed_path")))
            except (OSError, UnicodeError, ValueError) as exc:
                errors.append(f"{book_id}: {str(exc)[:240] or exc.__class__.__name__}")
                continue
            if existed:
                updated += 1
            else:
                created += 1
        self.write_base()
        return CatalogSyncResult(
            total=len(books),
            created=created,
            updated=updated,
            failed=len(errors),
            errors=tuple(errors),
        )

    def write_base(self) -> Path:
        self._validate_vault_root()
        with self._library_directory(create=True) as (directory, directory_fd):
            self._write_atomically(
                directory_fd,
                self.paths.catalog_base.name,
                self._base_content(),
            )
            return directory / self.paths.catalog_base.name

    @staticmethod
    def _preview(value: object) -> str:
        if not isinstance(value, str) or not value.strip():
            return ""
        try:
            with Path(value).open("r", encoding="utf-8", errors="replace") as source:
                return source.read(4000)
        except OSError:
            return ""

    @staticmethod
    def _base_content() -> str:
        return """filters:
  and:
    - 'file.inFolder("书库/50-书目卡片")'
    - 'file.ext == "md"'
properties:
  title:
    displayName: 书名
  author:
    displayName: 作者
  primary_category:
    displayName: 主分类
  custom_categories:
    displayName: 自定义分类
  library_status:
    displayName: 书库状态
  ocr_status:
    displayName: OCR 状态
  source_link:
    displayName: 原始书籍
  parsed_link:
    displayName: 解析正文
  ocr_report_link:
    displayName: OCR 报告
views:
  - type: table
    name: "按主分类"
    groupBy:
      property: note.primary_category
      direction: ASC
    order:
      - title
      - author
      - custom_categories
      - ocr_status
      - source_link
      - parsed_link
      - ocr_report_link
  - type: table
    name: "全部书籍"
    order:
      - title
      - author
      - primary_category
      - custom_categories
      - library_status
      - ocr_status
      - source_link
      - parsed_link
      - ocr_report_link
  - type: table
    name: "待 OCR"
    filters:
      or:
        - 'library_status == "needs_ocr"'
        - 'ocr_status == "queued"'
        - 'ocr_status == "running"'
        - 'ocr_status == "paused"'
    order:
      - title
      - author
      - ocr_status
      - source_link
  - type: table
    name: "OCR 有警告"
    filters:
      and:
        - 'ocr_status == "warning"'
    order:
      - title
      - author
      - ocr_report_link
      - source_link
"""

    def _catalog_directory(self, *, create: bool):
        return _managed_directory_beneath(
            self.paths.vault,
            self.paths.catalog_cards,
            "catalog cards",
            create=create,
            create_root=False,
            root_label="vault root",
            expected_root_identity=self._vault_root_identity,
        )

    def _library_directory(self, *, create: bool):
        return _managed_directory_beneath(
            self.paths.vault,
            self.paths.library,
            "library",
            create=create,
            create_root=False,
            root_label="vault root",
            expected_root_identity=self._vault_root_identity,
        )

    @staticmethod
    def _matching_card_names(directory_fd: int, book_id: str) -> list[str]:
        suffix = f"-{book_id}.md"
        return sorted(
            name
            for name in os.listdir(directory_fd)
            if isinstance(name, str) and name.endswith(suffix)
        )

    def _card_exists(self, book_id: str) -> bool:
        with self._catalog_directory(create=True) as (_, directory_fd):
            return bool(self._matching_card_names(directory_fd, book_id))

    def _card_name(self, title: str, book_id: str, directory_fd: int) -> str:
        matches = self._matching_card_names(directory_fd, book_id)
        if len(matches) > 1:
            raise ValueError(f"multiple catalog cards exist for book_id {book_id}")
        if matches:
            return matches[0]
        suffix = f"-{book_id}.md"
        title_budget = _MAX_FILENAME_BYTES - len(suffix.encode("utf-8"))
        sanitized = _UNSAFE_FILENAME.sub("_", title).strip(" .") or "未命名书籍"
        safe_title = sanitized.encode("utf-8")[:title_budget].decode(
            "utf-8", errors="ignore"
        ).strip(" .") or "未命名书籍"
        return f"{safe_title}{suffix}"

    @staticmethod
    def _read_card(directory_fd: int, name: str) -> str | None:
        try:
            descriptor = os.open(
                name,
                os.O_RDONLY
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=directory_fd,
            )
        except FileNotFoundError:
            return None
        except OSError as exc:
            raise ValueError("existing catalog card cannot be opened safely") from exc
        try:
            info = os.fstat(descriptor)
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                raise ValueError("existing catalog card must be a regular file")
            with os.fdopen(descriptor, "r", encoding="utf-8") as source:
                descriptor = -1
                return source.read()
        finally:
            if descriptor >= 0:
                os.close(descriptor)

    def _vault_link(self, value: object) -> str:
        if not isinstance(value, str) or not value.strip():
            return ""
        path = Path(value)
        try:
            relative = path.relative_to(self.paths.vault)
        except ValueError:
            return ""
        return f"[[{relative.as_posix()}]]"

    def _render(
        self,
        book: Mapping[str, Any],
        *,
        primary: str,
        custom: tuple[str, ...],
    ) -> str:
        book_id = str(book["book_id"])
        title = str(book["title"])
        source_link = self._vault_link(book.get("original_path"))
        parsed_link = self._vault_link(book.get("parsed_path"))
        report = self.paths.ocr_reports / f"{book_id}-OCR处理报告.md"
        try:
            report_info = report.lstat()
            report_exists = stat.S_ISREG(report_info.st_mode) and not stat.S_ISLNK(
                report_info.st_mode
            )
        except OSError:
            report_exists = False
        report_link = self._vault_link(str(report)) if report_exists else ""
        job = self.database.get_ocr_job(book_id)
        library_status = str(book.get("status") or "unknown")
        if report_link and self._report_has_warnings(report):
            ocr_status = "warning"
        elif job:
            ocr_status = str(job.get("status"))
        else:
            ocr_status = "needs_ocr" if library_status == "needs_ocr" else "not_required"
        custom_yaml = "[]" if not custom else "\n" + "\n".join(
            f"  - {_yaml_string(item)}" for item in custom
        )
        lines = [
            "---",
            f"book_id: {_yaml_string(book_id)}",
            f"title: {_yaml_string(title)}",
            f"author: {_yaml_string(book.get('author') or '')}",
            f"primary_category: {_yaml_string(primary)}",
            f"custom_categories: {custom_yaml}" if not custom else f"custom_categories:{custom_yaml}",
            f"source_format: {_yaml_string(book.get('source_format') or '')}",
            f"library_status: {_yaml_string(library_status)}",
            f"ocr_status: {_yaml_string(ocr_status)}",
            f"source_link: {_yaml_string(source_link)}",
            f"parsed_link: {_yaml_string(parsed_link)}",
            f"ocr_report_link: {_yaml_string(report_link)}",
            f"created_at: {_yaml_string(book.get('created_at') or '')}",
            f"updated_at: {_yaml_string(book.get('updated_at') or '')}",
            "---",
            "",
            f"# {title}",
            "",
            f"- {source_link[:-2] + '|打开原始书籍]]' if source_link else '原始书籍：暂无'}",
            f"- {parsed_link[:-2] + '|打开解析正文]]' if parsed_link else '解析正文：暂无'}",
            f"- {report_link[:-2] + '|打开 OCR 报告]]' if report_link else 'OCR 报告：暂无'}",
            "",
        ]
        return "\n".join(lines)

    @staticmethod
    def _report_has_warnings(report: Path) -> bool:
        try:
            with report.open("r", encoding="utf-8", errors="replace") as source:
                header = source.read(4096)
        except OSError:
            return True
        match = re.search(r"(?m)^skipped_pages:\s*(\d+)\s*$", header)
        return match is None or int(match.group(1)) > 0

    @staticmethod
    def _write_atomically(directory_fd: int, name: str, content: str) -> None:
        temporary_name = f".{name}.{secrets.token_hex(12)}.tmp"
        descriptor: int | None = None
        try:
            descriptor = os.open(
                temporary_name,
                os.O_CREAT
                | os.O_EXCL
                | os.O_WRONLY
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0),
                0o600,
                dir_fd=directory_fd,
            )
            with os.fdopen(descriptor, "w", encoding="utf-8") as temporary:
                descriptor = None
                temporary.write(content)
                temporary.flush()
                os.fsync(temporary.fileno())
            os.replace(
                temporary_name,
                name,
                src_dir_fd=directory_fd,
                dst_dir_fd=directory_fd,
            )
            temporary_name = ""
        finally:
            if descriptor is not None:
                os.close(descriptor)
            if temporary_name:
                try:
                    os.unlink(temporary_name, dir_fd=directory_fd)
                except FileNotFoundError:
                    pass


__all__ = ["CatalogService", "CatalogSyncResult", "classify_book"]
