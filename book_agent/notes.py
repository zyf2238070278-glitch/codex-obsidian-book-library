from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
import errno
import os
from pathlib import Path
import re
import secrets
from typing import Callable

from book_agent.config import AppPaths
from book_agent.markdown import markdown_literal, visible_text
from book_agent.storage import Database
from book_agent.vault import VaultManager, _secure_create_open_flags


@dataclass(frozen=True)
class SavedNote:
    path: str
    wiki_link: str


_UNSAFE_FILENAME = re.compile(r'[<>:"/\\|?*\[\]#^\x00-\x1f\x7f]')
_SAFE_CITATION_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}\Z")
_TEMP_PREFIX = ".note-"


def _safe_title(title: str) -> str:
    cleaned = _UNSAFE_FILENAME.sub("-", title)
    return visible_text(cleaned, single_line=True).strip(" .-")


def _verified_citation_link(hit: object) -> str:
    book_id = str(getattr(hit, "book_id"))
    passage_id = str(getattr(hit, "passage_id"))
    markdown_path = str(getattr(hit, "markdown_path"))
    anchor = str(getattr(hit, "anchor"))
    expected_path = f"书库/20-解析文本/{book_id}/正文.md"
    if (
        _SAFE_CITATION_ID.fullmatch(book_id) is None
        or _SAFE_CITATION_ID.fullmatch(passage_id) is None
        or markdown_path != expected_path
        or anchor != passage_id
    ):
        raise ValueError("Unverified internal citation target")
    return f"[[{expected_path}#^{passage_id}]]"


class NoteService:
    def __init__(
        self,
        paths: AppPaths,
        database: Database,
        clock: Callable[[], datetime] = datetime.now,
    ) -> None:
        self.paths = paths
        self.database = database
        self.clock = clock

    def save(
        self,
        title: str,
        markdown: str,
        passage_ids: Sequence[str],
    ) -> SavedNote:
        if not isinstance(title, str) or not title.strip():
            raise ValueError("title must be a non-blank string")
        if not isinstance(markdown, str) or not markdown.strip():
            raise ValueError("markdown must be a non-blank string")
        if isinstance(passage_ids, (str, bytes)) or not isinstance(
            passage_ids, Sequence
        ):
            raise ValueError("passage_ids must be a non-string sequence")
        if not passage_ids:
            raise ValueError("passage_ids must contain at least one ID")

        normalized_title = title.strip()
        normalized_markdown = markdown.strip()
        safe_title = _safe_title(normalized_title)
        if not safe_title:
            raise ValueError("title does not contain a safe filename character")

        requested: list[str] = []
        seen: set[str] = set()
        for passage_id in passage_ids:
            if not isinstance(passage_id, str) or not passage_id.strip():
                raise ValueError("each passage ID must be a non-blank string")
            if passage_id not in seen:
                seen.add(passage_id)
                requested.append(passage_id)

        returned = self.database.get_passages(requested)
        by_id = {hit.passage_id: hit for hit in returned}
        unknown = [passage_id for passage_id in requested if passage_id not in by_id]
        if unknown:
            raise ValueError(
                "Unknown or unavailable passage IDs: " + ", ".join(unknown)
            )
        hits = [by_id[passage_id] for passage_id in requested]
        citations: list[str] = []
        for hit in hits:
            internal_link = _verified_citation_link(hit)
            location = (
                markdown_literal(hit.section, single_line=True)
                if hit.section
                else hit.passage_id
            )
            if hit.page_start is not None:
                page = str(hit.page_start)
                if hit.page_end is not None and hit.page_end != hit.page_start:
                    page += f"–{hit.page_end}"
                location += f"，PDF 页 {page}"
            citations.append(
                f"- 《{markdown_literal(hit.title, single_line=True)}》：{location} "
                f"{internal_link}"
            )
        rendered_citations = "\n".join(citations)
        content = (
            "---\n"
            "source_type: ai_generated\n"
            "index_for_evidence: false\n"
            "created_by: codex-book-agent\n"
            "---\n\n"
            f"# {markdown_literal(normalized_title, single_line=True)}\n\n"
            f"{normalized_markdown}\n\n"
            "## 原文依据\n\n"
            f"{rendered_citations}\n"
        )

        timestamp = self.clock().strftime("%Y%m%d-%H%M%S")
        manager = VaultManager(self.paths)
        manager.ensure_layout()
        directory_fd: int | None = None
        notes_directory = self.paths.notes
        try:
            with manager._managed_directory(
                self.paths.notes,
                "notes",
                create=False,
            ) as (notes_directory, verified_fd):
                directory_fd = os.dup(verified_fd)
        except BaseException:
            if directory_fd is not None:
                try:
                    os.close(directory_fd)
                except OSError:
                    pass
            raise
        destination = self._publish(
            content,
            safe_title,
            timestamp,
            directory_fd,
            notes_directory,
        )
        relative = destination.relative_to(self.paths.vault).with_suffix("")
        return SavedNote(
            path=str(Path(destination).absolute()),
            wiki_link=f"[[{relative.as_posix()}]]",
        )

    def _publish(
        self,
        content: str,
        safe_title: str,
        timestamp: str,
        directory_fd: int,
        notes_directory: Path,
    ) -> Path:
        temp_name: str | None = None
        published: Path | None = None
        primary_error: BaseException | None = None
        try:
            temp_name, temp_fd = self._reserve_temp(directory_fd)
            write_error: BaseException | None = None
            try:
                self._write_complete(temp_fd, content.encode("utf-8"))
            except BaseException as exc:
                write_error = exc
            try:
                os.close(temp_fd)
            except OSError:
                if write_error is None:
                    raise
            if write_error is not None:
                raise write_error

            try:
                name_max = int(os.fpathconf(directory_fd, "PC_NAME_MAX"))
            except (OSError, ValueError) as exc:
                raise RuntimeError(
                    "Unable to determine the notes filesystem filename limit"
                ) from exc
            if name_max <= 0:
                raise RuntimeError("The notes filesystem reported an invalid filename limit")

            for candidate in self._candidate_names(
                safe_title,
                timestamp,
                name_max,
            ):
                try:
                    os.link(
                        temp_name,
                        candidate,
                        src_dir_fd=directory_fd,
                        dst_dir_fd=directory_fd,
                        follow_symlinks=False,
                    )
                except OSError as exc:
                    if exc.errno == errno.EEXIST:
                        continue
                    raise RuntimeError(
                        "Atomic hard-link publication failed; the notes filesystem "
                        "may not support hard links or cross-filesystem linking"
                    ) from exc
                published = notes_directory / candidate
                break
        except BaseException as exc:
            primary_error = exc

        cleanup_error: OSError | None = None
        try:
            if temp_name is not None:
                os.unlink(temp_name, dir_fd=directory_fd)
        except FileNotFoundError:
            pass
        except OSError as exc:
            cleanup_error = exc
        try:
            os.close(directory_fd)
        except OSError as exc:
            if cleanup_error is None:
                cleanup_error = exc

        if primary_error is not None:
            raise primary_error
        if published is not None:
            return published
        if cleanup_error is not None:
            raise RuntimeError("Unable to clean up an unpublished temporary note") from cleanup_error
        raise RuntimeError("Unable to choose a unique note filename")

    @staticmethod
    def _reserve_temp(directory_fd: int) -> tuple[str, int]:
        for _ in range(128):
            name = f"{_TEMP_PREFIX}{secrets.token_hex(12)}"
            try:
                return name, os.open(
                    name,
                    _secure_create_open_flags(),
                    0o600,
                    dir_fd=directory_fd,
                )
            except FileExistsError:
                continue
        raise RuntimeError("Unable to reserve a unique temporary note file")

    @staticmethod
    def _write_complete(file_descriptor: int, payload: bytes) -> None:
        offset = 0
        while offset < len(payload):
            written = os.write(file_descriptor, payload[offset:])
            if written <= 0:
                raise OSError("Incomplete temporary note write")
            offset += written
        os.fsync(file_descriptor)

    @staticmethod
    def _candidate_names(safe_title: str, timestamp: str, name_max: int):
        yield NoteService._fit_filename(safe_title, ".md", name_max)
        yield NoteService._fit_filename(
            safe_title,
            f"-{timestamp}.md",
            name_max,
        )
        counter = 2
        while True:
            yield NoteService._fit_filename(
                safe_title,
                f"-{timestamp}-{counter}.md",
                name_max,
            )
            counter += 1

    @staticmethod
    def _fit_filename(base: str, suffix: str, name_max: int) -> str:
        budget = name_max - len(suffix.encode("utf-8"))
        if budget <= 0:
            raise RuntimeError(
                "The notes filesystem filename limit is too small for note suffixes"
            )
        encoded = base.encode("utf-8")
        if len(encoded) > budget:
            fitted = encoded[:budget].decode("utf-8", errors="ignore")
        else:
            fitted = base
        fitted = fitted.rstrip(" .-")
        if not fitted:
            raise RuntimeError(
                "The note title cannot fit within the filesystem filename limit"
            )
        return f"{fitted}{suffix}"
