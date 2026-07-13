import errno
import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from contextlib import contextmanager
from dataclasses import FrozenInstanceError
from pathlib import Path
from threading import Event, Lock

import fitz
import numpy as np
import pytest

import book_agent.importer as importer_module
from book_agent.config import AppPaths
from book_agent.embeddings import NullEmbeddingProvider, decode_vector, encode_vector
from book_agent.importer import ImportResult, ImportService, sha256_file
from book_agent.storage import Database


class _ReadyEmbeddingProvider:
    available = True

    def __init__(self) -> None:
        self.received: list[str] = []

    def embed_query(self, text: str) -> np.ndarray:
        return np.array([1.0, 2.0, 3.0], dtype=np.float32)

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        self.received = list(texts)
        return np.array(
            [[ordinal, ordinal + 0.5, ordinal + 1.0] for ordinal, _ in enumerate(texts)],
            dtype=np.float64,
        )


class _WrongCountEmbeddingProvider(_ReadyEmbeddingProvider):
    def embed_passages(self, texts: list[str]) -> np.ndarray:
        self.received = list(texts)
        return np.empty((0, 3), dtype=np.float32)


class _FailingEmbeddingProvider(_ReadyEmbeddingProvider):
    def embed_passages(self, texts: list[str]) -> np.ndarray:
        self.received = list(texts)
        raise RuntimeError("模型暂时不可用")


class _InvalidVectorEmbeddingProvider(_ReadyEmbeddingProvider):
    def __init__(self, case: str) -> None:
        super().__init__()
        self.case = case

    def embed_passages(self, texts: list[str]):
        self.received = list(texts)
        count = len(texts)
        if self.case == "empty":
            return np.empty((count, 0), dtype=np.float32)
        if self.case == "nan":
            vectors = np.ones((count, 3), dtype=np.float32)
            vectors[0, 0] = np.nan
            return vectors
        if self.case == "inf":
            vectors = np.ones((count, 3), dtype=np.float32)
            vectors[0, 0] = np.inf
            return vectors
        if self.case == "two-dimensional-row":
            return [np.ones((1, 3), dtype=np.float32) for _ in texts]
        if self.case == "inconsistent":
            return [
                np.ones(2 + ordinal, dtype=np.float32)
                for ordinal, _ in enumerate(texts)
            ]
        raise AssertionError(f"unknown vector case: {self.case}")


class _InterruptingEmbeddingProvider(_ReadyEmbeddingProvider):
    def embed_passages(self, texts: list[str]) -> np.ndarray:
        self.received = list(texts)
        raise KeyboardInterrupt("operator cancelled embedding")


@pytest.fixture
def app(tmp_path: Path) -> tuple[AppPaths, Database]:
    paths = AppPaths.from_root(tmp_path / "app")
    database = Database(paths.database)
    database.initialize()
    return paths, database


def _write_txt(path: Path, marker: str = "库存周期") -> Path:
    path.write_text(
        f"第一段讨论{marker}与产业需求。\n\n第二段记录风险与机会。",
        encoding="utf-8",
    )
    return path


def _write_multi_passage_txt(path: Path) -> Path:
    path.write_text(
        "第一段" + "甲" * 1600 + "\n\n" + "第二段" + "乙" * 1600,
        encoding="utf-8",
    )
    return path


def _write_textless_pdf(path: Path) -> Path:
    document = fitz.open()
    try:
        document.new_page()
        document.new_page()
        document.save(path)
    finally:
        document.close()
    return path


def test_import_rejects_symlinked_parsed_book_directory_without_external_write(
    app: tuple[AppPaths, Database],
    tmp_path: Path,
) -> None:
    paths, database = app
    source = _write_txt(tmp_path / "symlinked-render.txt")
    book_id = sha256_file(source)[:24]
    external_directory = tmp_path / "outside-render"
    external_directory.mkdir()
    paths.parsed.mkdir(parents=True)
    (paths.parsed / book_id).symlink_to(
        external_directory,
        target_is_directory=True,
    )

    result = ImportService(
        paths,
        database,
        NullEmbeddingProvider(),
    ).import_book(source)

    assert result.status == "failed"
    assert "symlink" in result.message.lower() or "符号链接" in result.message
    assert not (external_directory / "正文.md").exists()


def test_vector_codec_forces_contiguous_float32_and_returns_an_owned_copy() -> None:
    source = np.arange(20, dtype=np.float64).reshape(4, 5)[:, ::2]
    assert not source.flags.c_contiguous

    payload = encode_vector(source)
    decoded = decode_vector(payload)

    assert isinstance(payload, bytes)
    assert decoded.dtype == np.float32
    assert decoded.flags.c_contiguous
    assert decoded.flags.owndata
    assert decoded.flags.writeable
    np.testing.assert_array_equal(decoded, source.astype(np.float32).ravel())


def test_null_embedding_provider_is_explicitly_unavailable() -> None:
    provider = NullEmbeddingProvider()

    assert provider.available is False
    with pytest.raises(RuntimeError, match="语义|模型"):
        provider.embed_query("查询")
    with pytest.raises(RuntimeError, match="语义|模型"):
        provider.embed_passages(["段落"])


def test_import_result_is_frozen_and_json_compatible() -> None:
    result = ImportResult(
        book_id="abc123",
        status="keyword_only",
        source_format="txt",
        original_path="/vault/original.txt",
        parsed_path=None,
        passage_count=2,
        message="仅关键词检索",
    )

    serialized = result.to_dict()

    assert type(serialized) is dict
    assert json.loads(json.dumps(serialized, ensure_ascii=False)) == serialized
    with pytest.raises(FrozenInstanceError):
        result.status = "ready"  # type: ignore[misc]


def test_sha256_file_is_stable_for_streamed_content(tmp_path: Path) -> None:
    path = tmp_path / "large.txt"
    content = b"abc123" * 300_000
    path.write_bytes(content)

    expected = hashlib.sha256(content).hexdigest()

    assert sha256_file(path) == expected
    assert sha256_file(path) == expected


def test_txt_import_is_keyword_searchable_and_a_second_import_is_duplicate(
    app: tuple[AppPaths, Database], tmp_path: Path
) -> None:
    paths, database = app
    source = _write_txt(tmp_path / "市场笔记.TXT")
    service = ImportService(paths, database, NullEmbeddingProvider())

    first = service.import_book(source, author="研究员")

    assert first.status == "keyword_only"
    assert first.source_format == "txt"
    assert first.book_id == sha256_file(source)[:24]
    assert first.passage_count > 0
    assert Path(first.original_path).is_absolute()
    assert Path(first.original_path).is_file()
    assert first.parsed_path is not None
    assert Path(first.parsed_path).is_file()
    book = database.get_book(first.book_id)
    assert book is not None
    assert book["status"] == "keyword_only"
    assert book["original_path"] == first.original_path
    assert book["parsed_path"] == first.parsed_path
    hits = database.keyword_search("库存周期", 5)
    assert hits
    assert hits[0].markdown_path == f"书库/20-解析文本/{first.book_id}/正文.md"
    parsed_content = Path(first.parsed_path).read_text(encoding="utf-8")
    assert f"^{hits[0].anchor}" in parsed_content
    original_count = len(list(paths.originals.iterdir()))

    duplicate = service.import_book(source)

    assert duplicate.status == "duplicate"
    assert duplicate.book_id == first.book_id
    assert duplicate.original_path == first.original_path
    assert duplicate.parsed_path == first.parsed_path
    assert duplicate.passage_count == first.passage_count
    assert len(list(paths.originals.iterdir())) == original_count
    assert len(database.list_books()) == 1


def test_copied_bytes_are_authoritative_when_source_changes_after_preflight_hash(
    app: tuple[AppPaths, Database],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths, database = app
    source = _write_txt(tmp_path / "changing.txt", marker="旧内容标记")
    service = ImportService(paths, database, NullEmbeddingProvider())
    real_import_original = service.vault.import_original

    def replace_then_copy(source_path: Path) -> Path:
        _write_txt(source, marker="副本权威内容")
        return real_import_original(source_path)

    monkeypatch.setattr(service.vault, "import_original", replace_then_copy)

    result = service.import_book(source)

    original = Path(result.original_path)
    authoritative_hash = hashlib.sha256(original.read_bytes()).hexdigest()
    book = database.get_book(result.book_id)
    assert result.book_id == authoritative_hash[:24]
    assert book["content_sha256"] == authoritative_hash
    assert Path(book["original_path"]) == original
    assert "副本权威内容" in Path(result.parsed_path).read_text(encoding="utf-8")
    assert database.keyword_search("副本权威内容", 5)
    assert database.keyword_search("旧内容标记", 5) == []


def test_hash_drift_to_an_existing_book_removes_the_new_copy_as_duplicate(
    app: tuple[AppPaths, Database],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths, database = app
    existing_source = _write_txt(tmp_path / "existing.txt", marker="权威重复内容")
    service = ImportService(paths, database, NullEmbeddingProvider())
    existing = service.import_book(existing_source)
    original_count = len(list(paths.originals.iterdir()))
    changing_source = _write_txt(tmp_path / "changing.txt", marker="预检旧内容")
    real_import_original = service.vault.import_original

    def replace_with_existing_then_copy(source_path: Path) -> Path:
        changing_source.write_bytes(existing_source.read_bytes())
        return real_import_original(source_path)

    monkeypatch.setattr(
        service.vault, "import_original", replace_with_existing_then_copy
    )

    duplicate = service.import_book(changing_source)

    assert duplicate.status == "duplicate"
    assert duplicate.book_id == existing.book_id
    assert duplicate.original_path == existing.original_path
    assert duplicate.parsed_path == existing.parsed_path
    assert duplicate.passage_count == existing.passage_count
    assert len(list(paths.originals.iterdir())) == original_count
    assert len(database.list_books()) == 1


def test_hash_drift_relocks_authoritative_content_before_concurrent_registration(
    app: tuple[AppPaths, Database],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths, first_database = app
    source = _write_txt(tmp_path / "drifting-concurrent.txt", marker="预检内容甲")
    replacement = _write_txt(tmp_path / "replacement.txt", marker="权威内容乙")
    preflight_hash = sha256_file(source)
    authoritative_hash = sha256_file(replacement)
    assert preflight_hash != authoritative_hash

    second_database = Database(paths.database)
    first_service = ImportService(paths, first_database, NullEmbeddingProvider())
    second_service = ImportService(paths, second_database, NullEmbeddingProvider())
    real_import_original = first_service.vault.import_original
    real_first_find = first_database.find_book_by_hash
    real_second_find = second_database.find_book_by_hash
    real_first_lock = first_service._book_lock
    source_replaced = Event()
    stale_lock_lookup_entered = Event()
    second_authoritative_lookup_done = Event()
    held_lock_guard = Lock()
    held_lock_id: str | None = None

    @contextmanager
    def track_first_lock(book_id: str):
        nonlocal held_lock_id
        with real_first_lock(book_id):
            with held_lock_guard:
                held_lock_id = book_id
            try:
                yield
            finally:
                with held_lock_guard:
                    held_lock_id = None

    def replace_then_copy(source_path: Path) -> Path:
        source.write_bytes(replacement.read_bytes())
        copied = real_import_original(source_path)
        source_replaced.set()
        return copied

    def find_from_first(content_hash: str):
        with held_lock_guard:
            current_lock_id = held_lock_id
        if (
            content_hash == authoritative_hash
            and current_lock_id == preflight_hash[:24]
        ):
            result = real_first_find(content_hash)
            assert result is None
            stale_lock_lookup_entered.set()
            if not second_authoritative_lookup_done.wait(timeout=5):
                raise TimeoutError("second authoritative lookup did not complete")
            return result
        return real_first_find(content_hash)

    def find_from_second(content_hash: str):
        if content_hash == authoritative_hash:
            # The broken implementation reaches the B lookup while still holding
            # A.  A corrected implementation serializes on B, so this wait may
            # legitimately time out before the only B owner performs the lookup.
            stale_lock_lookup_entered.wait(timeout=0.5)
            result = real_second_find(content_hash)
            second_authoritative_lookup_done.set()
            return result
        return real_second_find(content_hash)

    monkeypatch.setattr(first_service, "_book_lock", track_first_lock)
    monkeypatch.setattr(first_service.vault, "import_original", replace_then_copy)
    monkeypatch.setattr(first_database, "find_book_by_hash", find_from_first)
    monkeypatch.setattr(second_database, "find_book_by_hash", find_from_second)

    with ThreadPoolExecutor(max_workers=2) as pool:
        first_future = pool.submit(first_service.import_book, source)
        assert source_replaced.wait(timeout=5)
        second_future = pool.submit(second_service.import_book, source)
        first = first_future.result(timeout=10)
        second = second_future.result(timeout=10)

    assert {first.status, second.status} == {"keyword_only", "duplicate"}
    assert first.book_id == second.book_id == authoritative_hash[:24]
    assert len(first_database.list_books()) == 1
    assert len(list(paths.originals.iterdir())) == 1


def test_continuously_changing_source_stops_without_database_or_file_residue(
    app: tuple[AppPaths, Database],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths, database = app
    source = _write_txt(tmp_path / "always-changing.txt", marker="初始内容")
    service = ImportService(paths, database, NullEmbeddingProvider())
    real_import_original = service.vault.import_original
    copy_calls = 0

    def change_before_every_copy(source_path: Path) -> Path:
        nonlocal copy_calls
        copy_calls += 1
        _write_txt(source, marker=f"第{copy_calls}次变化后的内容")
        return real_import_original(source_path)

    monkeypatch.setattr(service.vault, "import_original", change_before_every_copy)

    with pytest.raises(ValueError, match="持续.*变化|多次.*变化"):
        service.import_book(source)

    assert 1 < copy_calls <= 4
    assert database.list_books() == []
    assert list(paths.originals.iterdir()) == []
    assert list(paths.inbox.iterdir()) == []


def test_post_copy_hash_failure_reraises_and_removes_the_unregistered_original(
    app: tuple[AppPaths, Database],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths, database = app
    source = _write_txt(tmp_path / "rehash-failure.txt")
    real_sha256_file = importer_module.sha256_file
    injected_error = OSError("post-copy rehash unavailable")
    calls = 0

    def fail_second_hash(path: str | Path) -> str:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise injected_error
        return real_sha256_file(path)

    monkeypatch.setattr(importer_module, "sha256_file", fail_second_hash)

    with pytest.raises(OSError) as caught:
        ImportService(paths, database, NullEmbeddingProvider()).import_book(source)

    assert caught.value is injected_error
    assert calls == 2
    assert database.list_books() == []
    assert paths.originals.is_dir()
    assert list(paths.originals.iterdir()) == []


def test_second_duplicate_lookup_failure_cleans_drifted_unregistered_original(
    app: tuple[AppPaths, Database],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths, database = app
    source = _write_txt(tmp_path / "lookup-failure.txt", marker="预检内容")
    service = ImportService(paths, database, NullEmbeddingProvider())
    real_import_original = service.vault.import_original
    real_find = database.find_book_by_hash
    injected_error = OSError("second duplicate lookup unavailable")
    find_calls = 0

    def replace_then_copy(source_path: Path) -> Path:
        _write_txt(source, marker="漂移后内容")
        return real_import_original(source_path)

    def fail_second_find(content_hash: str):
        nonlocal find_calls
        find_calls += 1
        if find_calls == 2:
            raise injected_error
        return real_find(content_hash)

    monkeypatch.setattr(service.vault, "import_original", replace_then_copy)
    monkeypatch.setattr(database, "find_book_by_hash", fail_second_find)

    with pytest.raises(OSError) as caught:
        service.import_book(source)

    assert caught.value is injected_error
    assert find_calls == 2
    assert database.list_books() == []
    assert paths.originals.is_dir()
    assert list(paths.originals.iterdir()) == []


def test_available_provider_marks_ready_and_persists_every_embedding(
    app: tuple[AppPaths, Database], tmp_path: Path
) -> None:
    paths, database = app
    source = _write_txt(tmp_path / "ready.txt")
    provider = _ReadyEmbeddingProvider()

    result = ImportService(paths, database, provider).import_book(source)

    assert result.status == "ready"
    assert database.get_book(result.book_id)["status"] == "ready"
    embedded = list(database.iter_embeddings([result.book_id]))
    assert len(embedded) == result.passage_count == len(provider.received)
    for ordinal, (hit, payload) in enumerate(embedded):
        assert hit.text == provider.received[ordinal]
        np.testing.assert_array_equal(
            decode_vector(payload),
            np.array([ordinal, ordinal + 0.5, ordinal + 1.0], dtype=np.float32),
        )


def test_keyword_only_reimport_with_available_provider_recovers_embeddings(
    app: tuple[AppPaths, Database], tmp_path: Path
) -> None:
    paths, database = app
    source = _write_txt(tmp_path / "recover-keyword-only.txt", marker="恢复语义索引")
    first = ImportService(paths, database, NullEmbeddingProvider()).import_book(source)
    assert first.parsed_path is not None
    Path(first.parsed_path).unlink()
    original_count = len(list(paths.originals.iterdir()))
    provider = _ReadyEmbeddingProvider()

    recovered = ImportService(paths, database, provider).import_book(source)

    assert first.status == "keyword_only"
    assert recovered.status == "ready"
    assert recovered.book_id == first.book_id
    assert recovered.original_path == first.original_path
    assert recovered.parsed_path == first.parsed_path
    assert Path(recovered.parsed_path).is_file()
    assert recovered.passage_count == first.passage_count == len(provider.received)
    assert database.get_book(first.book_id)["status"] == "ready"
    assert len(list(database.iter_embeddings([first.book_id]))) == first.passage_count
    assert len(list(paths.originals.iterdir())) == original_count
    assert len(database.list_books()) == 1


def test_failed_reimport_retries_pipeline_without_copying_original_again(
    app: tuple[AppPaths, Database],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths, database = app
    source = _write_txt(tmp_path / "recover-failed.txt", marker="失败后恢复")
    real_parse = importer_module.parse_document

    def fail_once(*args, **kwargs):
        raise RuntimeError("temporary parser failure")

    monkeypatch.setattr(importer_module, "parse_document", fail_once)
    first = ImportService(paths, database, NullEmbeddingProvider()).import_book(source)
    monkeypatch.setattr(importer_module, "parse_document", real_parse)
    original_count = len(list(paths.originals.iterdir()))

    recovered = ImportService(paths, database, NullEmbeddingProvider()).import_book(
        source
    )

    assert first.status == "failed"
    assert recovered.status == "keyword_only"
    assert recovered.book_id == first.book_id
    assert recovered.original_path == first.original_path
    assert recovered.passage_count > 0
    assert database.keyword_search("失败后恢复", 5)
    assert len(list(paths.originals.iterdir())) == original_count
    assert len(database.list_books()) == 1


def test_failed_reimport_restores_a_missing_managed_original_from_supplied_file(
    app: tuple[AppPaths, Database],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths, database = app
    source = _write_txt(tmp_path / "restore-original.txt", marker="恢复原书")
    real_parse = importer_module.parse_document

    def fail_once(*args, **kwargs):
        raise RuntimeError("temporary parser failure")

    monkeypatch.setattr(importer_module, "parse_document", fail_once)
    first = ImportService(paths, database, NullEmbeddingProvider()).import_book(source)
    missing_original = Path(first.original_path)
    missing_original.unlink()
    monkeypatch.setattr(importer_module, "parse_document", real_parse)

    recovered = ImportService(paths, database, NullEmbeddingProvider()).import_book(
        source
    )

    persisted = database.get_book(first.book_id)
    assert first.status == "failed"
    assert recovered.status == "keyword_only"
    assert recovered.book_id == first.book_id
    assert Path(recovered.original_path).is_file()
    assert Path(recovered.original_path).read_bytes() == source.read_bytes()
    assert persisted["original_path"] == recovered.original_path
    assert len(list(paths.originals.iterdir())) == 1
    assert len(database.list_books()) == 1


def test_processing_reimport_finishes_committed_passages_after_status_failure(
    app: tuple[AppPaths, Database],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths, database = app
    source = _write_txt(tmp_path / "recover-processing.txt", marker="处理中恢复")
    real_update = database.update_book_status

    def always_fail(*args, **kwargs) -> None:
        raise OSError("status database unavailable")

    monkeypatch.setattr(database, "update_book_status", always_fail)
    first = ImportService(paths, database, NullEmbeddingProvider()).import_book(source)
    monkeypatch.setattr(database, "update_book_status", real_update)
    original_count = len(list(paths.originals.iterdir()))

    recovered = ImportService(paths, database, NullEmbeddingProvider()).import_book(
        source
    )

    assert first.status == "processing"
    assert first.passage_count > 0
    assert recovered.status == "keyword_only"
    assert recovered.book_id == first.book_id
    assert recovered.original_path == first.original_path
    assert recovered.passage_count == first.passage_count
    assert database.get_book(first.book_id)["status"] == "keyword_only"
    assert database.keyword_search("处理中恢复", 5)
    assert len(list(paths.originals.iterdir())) == original_count
    assert len(database.list_books()) == 1


def test_same_hash_imports_are_serialized_until_the_first_status_is_final(
    app: tuple[AppPaths, Database],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths, database = app
    source = _write_txt(tmp_path / "concurrent.txt", marker="并发状态安全")
    real_parse = importer_module.parse_document
    first_parse_entered = Event()
    release_first_parse = Event()
    second_import_entered = Event()
    counter_lock = Lock()
    parse_calls = 0

    def hold_first_parse(*args, **kwargs):
        nonlocal parse_calls
        with counter_lock:
            parse_calls += 1
            call_number = parse_calls
        if call_number == 1:
            first_parse_entered.set()
            if not release_first_parse.wait(timeout=5):
                raise TimeoutError("test did not release first parse")
        return real_parse(*args, **kwargs)

    monkeypatch.setattr(importer_module, "parse_document", hold_first_parse)
    first_service = ImportService(paths, database, _ReadyEmbeddingProvider())
    second_service = ImportService(paths, database, _ReadyEmbeddingProvider())

    def import_second() -> ImportResult:
        second_import_entered.set()
        return second_service.import_book(source)

    with ThreadPoolExecutor(max_workers=2) as pool:
        first_future = pool.submit(first_service.import_book, source)
        assert first_parse_entered.wait(timeout=5)
        second_future = pool.submit(import_second)
        assert second_import_entered.wait(timeout=5)
        try:
            with pytest.raises(FutureTimeoutError):
                second_future.result(timeout=0.2)
        finally:
            release_first_parse.set()
        first = first_future.result(timeout=5)
        second = second_future.result(timeout=5)

    assert first.status == "ready"
    assert second.status == "duplicate"
    assert parse_calls == 1
    assert len(database.list_books()) == 1
    assert len(list(paths.originals.iterdir())) == 1


def test_book_lock_stays_anchored_when_lock_directory_is_swapped_for_symlink(
    app: tuple[AppPaths, Database],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths, database = app
    service = ImportService(paths, database, NullEmbeddingProvider())
    book_id = "a" * 24
    lock_directory = paths.database.parent / ".import-locks"
    displaced_directory = paths.database.parent / ".import-locks-displaced"
    external_directory = tmp_path / "external-lock-domain"
    external_directory.mkdir()
    real_open = os.open
    swapped = False

    def swap_before_lock_file_open(
        path,
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal swapped
        if (
            not swapped
            and flags & os.O_CREAT
            and os.fspath(path).endswith(f"{book_id}.lock")
        ):
            lock_directory.rename(displaced_directory)
            lock_directory.symlink_to(external_directory, target_is_directory=True)
            swapped = True
        return real_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(importer_module.os, "open", swap_before_lock_file_open)

    with service._book_lock(book_id):
        pass

    assert swapped is True
    assert list(external_directory.iterdir()) == []
    assert (displaced_directory / f"{book_id}.lock").is_file()


def test_book_lock_rejects_a_symlinked_data_parent_without_external_writes(
    tmp_path: Path,
) -> None:
    root = tmp_path / "app"
    root.mkdir()
    external_directory = tmp_path / "external-data"
    external_directory.mkdir()
    (root / "data").symlink_to(external_directory, target_is_directory=True)
    paths = AppPaths.from_root(root)
    service = ImportService(
        paths,
        Database(paths.database),
        NullEmbeddingProvider(),
    )

    with pytest.raises(ValueError, match="锁|安全|symlink|symbolic"):
        with service._book_lock("b" * 24):
            pass

    assert list(external_directory.iterdir()) == []


@pytest.mark.parametrize("failure", [None, RuntimeError, KeyboardInterrupt])
def test_book_lock_closes_every_opened_descriptor(
    app: tuple[AppPaths, Database],
    monkeypatch: pytest.MonkeyPatch,
    failure: type[BaseException] | None,
) -> None:
    paths, database = app
    service = ImportService(paths, database, NullEmbeddingProvider())
    real_open = os.open
    opened_descriptors: list[int] = []

    def track_open(
        path,
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        descriptor = real_open(path, flags, mode, dir_fd=dir_fd)
        opened_descriptors.append(descriptor)
        return descriptor

    monkeypatch.setattr(importer_module.os, "open", track_open)

    if failure is None:
        with service._book_lock("c" * 24):
            pass
    else:
        with pytest.raises(failure, match="injected lock body failure"):
            with service._book_lock("c" * 24):
                raise failure("injected lock body failure")

    assert opened_descriptors
    for descriptor in set(opened_descriptors):
        with pytest.raises(OSError) as caught:
            os.fstat(descriptor)
        assert caught.value.errno == errno.EBADF


@pytest.mark.parametrize(
    ("provider", "detail"),
    [
        (_WrongCountEmbeddingProvider(), "数量"),
        (_FailingEmbeddingProvider(), "模型暂时不可用"),
    ],
)
def test_embedding_failures_degrade_to_keyword_only_with_recovery_guidance(
    app: tuple[AppPaths, Database],
    tmp_path: Path,
    provider: _ReadyEmbeddingProvider,
    detail: str,
) -> None:
    paths, database = app
    source = _write_txt(tmp_path / f"fallback-{detail}.txt", marker="语义降级")

    result = ImportService(paths, database, provider).import_book(source)

    assert result.status == "keyword_only"
    assert "稍后" in result.message
    assert detail in result.message
    book = database.get_book(result.book_id)
    assert book["status"] == "keyword_only"
    assert detail in book["error"]
    assert database.keyword_search("语义降级", 5)
    assert list(database.iter_embeddings([result.book_id])) == []


@pytest.mark.parametrize(
    ("case", "detail"),
    [
        ("empty", "不能为空"),
        ("nan", "有限"),
        ("inf", "有限"),
        ("two-dimensional-row", "一维"),
        ("inconsistent", "维度一致"),
    ],
)
def test_invalid_vectors_degrade_before_any_embedding_is_written(
    app: tuple[AppPaths, Database],
    tmp_path: Path,
    case: str,
    detail: str,
) -> None:
    paths, database = app
    source = _write_multi_passage_txt(tmp_path / f"invalid-vector-{case}.txt")
    provider = _InvalidVectorEmbeddingProvider(case)

    result = ImportService(paths, database, provider).import_book(source)

    assert result.passage_count >= 2
    assert result.status == "keyword_only"
    assert "稍后" in result.message
    assert detail in result.message
    assert database.get_book(result.book_id)["status"] == "keyword_only"
    assert list(database.iter_embeddings([result.book_id])) == []


def test_textless_pdf_is_preserved_with_needs_ocr_status(
    app: tuple[AppPaths, Database], tmp_path: Path
) -> None:
    paths, database = app
    source = _write_textless_pdf(tmp_path / "scan.pdf")

    result = ImportService(paths, database, NullEmbeddingProvider()).import_book(source)

    assert result.status == "needs_ocr"
    assert result.source_format == "pdf"
    assert result.passage_count == 0
    assert result.parsed_path is None
    assert Path(result.original_path).is_file()
    book = database.get_book(result.book_id)
    assert book["status"] == "needs_ocr"
    assert "OCR" in book["error"]
    assert book["parsed_path"] is None
    assert database.count_passages(result.book_id) == 0


def test_invalid_utf8_is_failed_but_the_original_is_preserved(
    app: tuple[AppPaths, Database], tmp_path: Path
) -> None:
    paths, database = app
    source = tmp_path / "broken.txt"
    source.write_bytes(b"valid prefix\xff")

    result = ImportService(paths, database, NullEmbeddingProvider()).import_book(source)

    assert result.status == "failed"
    assert result.passage_count == 0
    assert Path(result.original_path).read_bytes() == source.read_bytes()
    book = database.get_book(result.book_id)
    assert book["status"] == "failed"
    assert "UTF-8" in book["error"]
    assert database.count_passages(result.book_id) == 0


@pytest.mark.parametrize("case", ["unsupported", "missing", "directory", "symlink"])
def test_invalid_sources_are_rejected_before_vault_or_database_writes(
    app: tuple[AppPaths, Database], tmp_path: Path, case: str
) -> None:
    paths, database = app
    if case == "unsupported":
        source = tmp_path / "book.docx"
        source.write_text("content", encoding="utf-8")
    elif case == "missing":
        source = tmp_path / "missing.txt"
    elif case == "directory":
        source = tmp_path / "folder.txt"
        source.mkdir()
    else:
        target = _write_txt(tmp_path / "target.txt")
        source = tmp_path / "link.txt"
        source.symlink_to(target)

    with pytest.raises(ValueError):
        ImportService(paths, database, NullEmbeddingProvider()).import_book(source)

    assert database.list_books() == []
    assert not paths.originals.exists()


@pytest.mark.parametrize("failure_point", ["render", "replace"])
def test_main_pipeline_failures_leave_no_partial_passages(
    app: tuple[AppPaths, Database],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_point: str,
) -> None:
    paths, database = app
    source = _write_txt(tmp_path / f"{failure_point}.txt")

    def fail(*args, **kwargs) -> None:
        raise OSError(f"{failure_point} injected failure")

    if failure_point == "render":
        monkeypatch.setattr(importer_module, "render_parsed_book", fail)
    else:
        monkeypatch.setattr(database, "replace_passages", fail)

    result = ImportService(paths, database, NullEmbeddingProvider()).import_book(source)

    assert result.status == "failed"
    assert failure_point in result.message
    assert Path(result.original_path).is_file()
    assert database.get_book(result.book_id)["status"] == "failed"
    assert database.count_passages(result.book_id) == 0


def test_unexpected_parser_exception_is_reported_as_failed(
    app: tuple[AppPaths, Database],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths, database = app
    source = _write_txt(tmp_path / "third-party.txt")

    def explode(*args, **kwargs):
        raise KeyError("third-party exploded")

    monkeypatch.setattr(importer_module, "parse_document", explode)

    result = ImportService(paths, database, NullEmbeddingProvider()).import_book(source)

    assert result.status == "failed"
    assert "third-party exploded" in result.message
    assert database.get_book(result.book_id)["status"] == "failed"
    assert database.count_passages(result.book_id) == 0


def test_final_status_failure_falls_back_to_failed_without_deleting_passages(
    app: tuple[AppPaths, Database],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths, database = app
    source = _write_txt(tmp_path / "status-retry.txt", marker="不可泄露证据")
    real_update = database.update_book_status
    attempts = 0

    def fail_once(*args, **kwargs) -> None:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise OSError("first status write failed")
        real_update(*args, **kwargs)

    monkeypatch.setattr(database, "update_book_status", fail_once)

    result = ImportService(paths, database, NullEmbeddingProvider()).import_book(source)

    assert attempts == 2
    assert result.status == "failed"
    assert "状态写入失败" in result.message
    assert database.get_book(result.book_id)["status"] == "failed"
    assert database.count_passages(result.book_id) == result.passage_count > 0
    assert database.keyword_search("不可泄露证据", 5) == []
    assert database.get_neighbors(result.book_id, 0, 10) == []


def test_double_status_failure_reports_actual_processing_state_and_hides_passages(
    app: tuple[AppPaths, Database],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths, database = app
    source = _write_txt(tmp_path / "status-stuck.txt", marker="处理中证据")
    attempts = 0

    def always_fail(*args, **kwargs) -> None:
        nonlocal attempts
        attempts += 1
        raise OSError("status database unavailable")

    monkeypatch.setattr(database, "update_book_status", always_fail)

    result = ImportService(paths, database, NullEmbeddingProvider()).import_book(source)

    assert attempts == 2
    assert database.get_book(result.book_id)["status"] == "processing"
    assert result.status == "processing"
    assert "状态写入失败" in result.message
    assert "无法写入" in result.message
    assert database.count_passages(result.book_id) == result.passage_count > 0
    assert database.keyword_search("处理中证据", 5) == []
    assert list(database.iter_embeddings([result.book_id])) == []


def test_keyboard_interrupt_marks_failed_hides_committed_passages_and_reraises(
    app: tuple[AppPaths, Database], tmp_path: Path
) -> None:
    paths, database = app
    source = _write_txt(tmp_path / "interrupted.txt", marker="中断前已提交")
    book_id = sha256_file(source)[:24]

    with pytest.raises(KeyboardInterrupt, match="operator cancelled"):
        ImportService(paths, database, _InterruptingEmbeddingProvider()).import_book(
            source
        )

    book = database.get_book(book_id)
    assert book["status"] == "failed"
    assert "中断" in book["error"]
    assert database.count_passages(book_id) > 0
    assert database.keyword_search("中断前已提交", 5) == []
    assert database.get_neighbors(book_id, 0, 10) == []


@pytest.mark.parametrize(
    "error_type",
    [RuntimeError, KeyboardInterrupt],
)
def test_create_book_failure_removes_the_just_copied_original_and_reraises(
    app: tuple[AppPaths, Database],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    error_type: type[BaseException],
) -> None:
    paths, database = app
    source = _write_txt(tmp_path / "orphan.txt")

    def fail_create(*args, **kwargs) -> None:
        raise error_type("create book injected failure")

    monkeypatch.setattr(database, "create_book", fail_create)

    with pytest.raises(error_type, match="create book injected failure"):
        ImportService(paths, database, NullEmbeddingProvider()).import_book(source)

    assert database.list_books() == []
    assert paths.originals.is_dir()
    assert list(paths.originals.iterdir()) == []
