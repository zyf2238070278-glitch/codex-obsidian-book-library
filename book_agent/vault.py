import errno
import os
import secrets
import shutil
import stat
from collections.abc import Iterator
from contextlib import contextmanager
from itertools import count
from pathlib import Path

from book_agent.config import AppPaths


_TEMP_PREFIX = ".import-"
_TEMP_RANDOM_BYTES = 12


def _required_open_flag(name: str) -> int:
    flag = getattr(os, name, None)
    if not isinstance(flag, int) or flag == 0:
        raise RuntimeError(
            f"This platform lacks the secure filesystem flag required for {name}"
        )
    return flag


def _secure_directory_open_flags() -> int:
    return (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | _required_open_flag("O_DIRECTORY")
        | _required_open_flag("O_NOFOLLOW")
    )


def _secure_source_open_flags() -> int:
    return (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | _required_open_flag("O_NOFOLLOW")
    )


def _secure_create_open_flags() -> int:
    return (
        os.O_CREAT
        | os.O_EXCL
        | os.O_WRONLY
        | getattr(os, "O_CLOEXEC", 0)
        | _required_open_flag("O_NOFOLLOW")
    )


def _absolute_path(path: Path, label: str) -> Path:
    try:
        return Path(os.path.abspath(os.fspath(path.expanduser())))
    except (OSError, RuntimeError, TypeError) as exc:
        raise ValueError(f"Managed path '{label}' is invalid: {path}") from exc


def _confined_path(
    configured: Path,
    label: str,
    root: Path,
    *,
    root_label: str = "project root",
) -> Path:
    try:
        expanded = configured.expanduser()
        if not expanded.is_absolute():
            expanded = root / expanded
        directory = Path(os.path.abspath(os.fspath(expanded)))
        directory.relative_to(root)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise ValueError(
            f"Managed directory '{label}' is not beneath {root_label}: {configured}"
        ) from exc
    return directory


def _open_directory_component(
    current_fd: int,
    current_path: Path,
    component: str,
    label: str,
    *,
    create: bool,
) -> int:
    if create:
        try:
            os.mkdir(component, mode=0o700, dir_fd=current_fd)
        except FileExistsError:
            pass
        except OSError as exc:
            raise ValueError(
                f"Managed directory '{label}' cannot be created safely: "
                f"{current_path / component}"
            ) from exc

    try:
        entry_info = os.stat(
            component,
            dir_fd=current_fd,
            follow_symlinks=False,
        )
    except OSError as exc:
        raise ValueError(
            f"Managed directory '{label}' is unavailable: {current_path / component}"
        ) from exc
    if stat.S_ISLNK(entry_info.st_mode):
        raise ValueError(
            f"Managed directory '{label}' contains a symlink: "
            f"{current_path / component}"
        )
    if not stat.S_ISDIR(entry_info.st_mode):
        raise ValueError(
            f"Managed directory '{label}' contains a non-directory: "
            f"{current_path / component}"
        )

    try:
        return os.open(
            component,
            _secure_directory_open_flags(),
            dir_fd=current_fd,
        )
    except OSError as exc:
        raise ValueError(
            f"Managed directory '{label}' cannot be opened safely without "
            f"following symlinks: {current_path / component}"
        ) from exc


def _open_absolute_directory(directory: Path, label: str, *, create: bool) -> int:
    anchor = Path(directory.anchor)
    if not anchor.is_absolute():
        raise ValueError(f"Managed directory '{label}' must be absolute: {directory}")
    try:
        current_fd = os.open(anchor, _secure_directory_open_flags())
    except OSError as exc:
        raise ValueError(
            f"Filesystem root cannot be opened safely for '{label}': {anchor}"
        ) from exc

    current_path = anchor
    try:
        for component in directory.relative_to(anchor).parts:
            next_fd = _open_directory_component(
                current_fd,
                current_path,
                component,
                label,
                create=create,
            )
            previous_fd = current_fd
            current_fd = next_fd
            os.close(previous_fd)
            current_path /= component
        return current_fd
    except BaseException:
        os.close(current_fd)
        raise


@contextmanager
def _managed_directory_beneath(
    root_path: Path,
    configured: Path,
    label: str,
    *,
    create: bool,
    root_label: str = "project root",
) -> Iterator[tuple[Path, int]]:
    root = _absolute_path(root_path, root_label)
    directory = _confined_path(
        configured,
        label,
        root,
        root_label=root_label,
    )
    root_fd = _open_absolute_directory(root, root_label, create=create)
    current_fd = root_fd
    current_path = root
    try:
        for component in directory.relative_to(root).parts:
            next_fd = _open_directory_component(
                current_fd,
                current_path,
                component,
                label,
                create=create,
            )
            previous_fd = current_fd
            current_fd = next_fd
            os.close(previous_fd)
            current_path /= component
        yield directory, current_fd
    finally:
        os.close(current_fd)


class VaultManager:
    def __init__(self, paths: AppPaths) -> None:
        self.paths = paths

    def ensure_layout(self) -> None:
        vault_directories = (
            ("inbox", self.paths.inbox),
            ("originals", self.paths.originals),
            ("parsed", self.paths.parsed),
            ("notes", self.paths.notes),
        )
        project_directories = (
            ("models", self.paths.models),
            ("database.parent", self.paths.database.parent),
        )
        for label, directory in vault_directories:
            with self._managed_directory(directory, label, create=True):
                pass
        for label, directory in project_directories:
            with self._project_directory(directory, label, create=True):
                pass

    def import_original(self, source: Path) -> Path:
        source_path, source_fd, source_info = self._open_source(source)
        try:
            with self._managed_directory(
                self.paths.inbox,
                "inbox",
                create=False,
            ) as (_, inbox_fd):
                with self._managed_directory(
                    self.paths.originals,
                    "originals",
                    create=False,
                ) as (originals, originals_fd):
                    temp_name, temp_fd, temp_info = self._reserve_temp(inbox_fd)
                    try:
                        self._copy_complete(
                            source_fd,
                            source_info,
                            temp_fd,
                            source,
                        )
                        final_name = self._link_final(
                            inbox_fd,
                            temp_name,
                            originals_fd,
                            source_path.name,
                        )
                        return originals / final_name
                    finally:
                        try:
                            os.close(temp_fd)
                        finally:
                            self._unlink_if_same(inbox_fd, temp_name, temp_info)
        finally:
            os.close(source_fd)

    def _open_source(self, source: Path) -> tuple[Path, int, os.stat_result]:
        try:
            expanded_source = Path(source).expanduser()
            source_path = Path(os.path.abspath(os.fspath(expanded_source)))
        except (OSError, RuntimeError, TypeError) as exc:
            raise ValueError(f"Source file is unavailable: {source}") from exc

        try:
            source_fd = os.open(source_path, _secure_source_open_flags())
        except OSError as exc:
            raise ValueError(f"Source file cannot be opened safely: {source}") from exc

        try:
            source_info = os.fstat(source_fd)
        except OSError as exc:
            os.close(source_fd)
            raise ValueError(f"Source file cannot be inspected: {source}") from exc
        if not stat.S_ISREG(source_info.st_mode):
            os.close(source_fd)
            raise ValueError(f"Source must be a regular file: {source}")

        return source_path, source_fd, source_info

    @contextmanager
    def _managed_directory(
        self,
        configured: Path,
        label: str,
        *,
        create: bool,
    ) -> Iterator[tuple[Path, int]]:
        with _managed_directory_beneath(
            self.paths.vault,
            configured,
            label,
            create=create,
            root_label="vault root",
        ) as managed:
            yield managed

    @contextmanager
    def _project_directory(
        self,
        configured: Path,
        label: str,
        *,
        create: bool,
    ) -> Iterator[tuple[Path, int]]:
        with _managed_directory_beneath(
            self.paths.root,
            configured,
            label,
            create=create,
        ) as managed:
            yield managed

    @staticmethod
    def _reserve_temp(directory_fd: int) -> tuple[str, int, os.stat_result]:
        for _ in range(128):
            temp_name = f"{_TEMP_PREFIX}{secrets.token_hex(_TEMP_RANDOM_BYTES)}"
            try:
                temp_fd = os.open(
                    temp_name,
                    _secure_create_open_flags(),
                    0o600,
                    dir_fd=directory_fd,
                )
            except FileExistsError:
                continue
            except OSError as exc:
                raise ValueError("Could not reserve private import temp file") from exc
            try:
                temp_info = os.fstat(temp_fd)
            except OSError as exc:
                os.close(temp_fd)
                try:
                    os.unlink(temp_name, dir_fd=directory_fd)
                except OSError:
                    pass
                raise ValueError("Could not inspect private import temp file") from exc
            return temp_name, temp_fd, temp_info

        raise ValueError("Could not reserve a unique private import temp file")

    @staticmethod
    def _copy_complete(
        source_fd: int,
        source_info: os.stat_result,
        temp_fd: int,
        source: Path,
    ) -> None:
        try:
            with os.fdopen(os.dup(source_fd), "rb") as source_stream:
                with os.fdopen(os.dup(temp_fd), "wb") as temp_stream:
                    shutil.copyfileobj(source_stream, temp_stream)
                    temp_stream.flush()
            os.fchmod(temp_fd, stat.S_IMODE(source_info.st_mode))
            os.utime(
                temp_fd,
                ns=(source_info.st_atime_ns, source_info.st_mtime_ns),
            )
            temp_after_copy = os.fstat(temp_fd)
            source_after_copy = os.fstat(source_fd)
            if temp_after_copy.st_size != source_info.st_size:
                raise ValueError("Copied file size does not match source")
            source_snapshot = (
                source_info.st_dev,
                source_info.st_ino,
                source_info.st_size,
                source_info.st_mtime_ns,
            )
            current_source = (
                source_after_copy.st_dev,
                source_after_copy.st_ino,
                source_after_copy.st_size,
                source_after_copy.st_mtime_ns,
            )
            if current_source != source_snapshot:
                raise ValueError("Source changed while it was being copied")
            os.fsync(temp_fd)
        except Exception as exc:
            raise ValueError(f"Could not copy source during atomic import: {source}") from exc

    @staticmethod
    def _link_final(
        inbox_fd: int,
        temp_name: str,
        originals_fd: int,
        source_name: str,
    ) -> str:
        try:
            name_max = os.fpathconf(originals_fd, "PC_NAME_MAX")
        except (OSError, ValueError) as exc:
            raise ValueError("Could not determine originals filename limit") from exc
        if name_max <= 0:
            raise ValueError("Originals filename limit is invalid")

        for index in count(1):
            candidate = VaultManager._candidate_name(source_name, index, name_max)
            try:
                os.link(
                    temp_name,
                    candidate,
                    src_dir_fd=inbox_fd,
                    dst_dir_fd=originals_fd,
                    follow_symlinks=False,
                )
            except FileExistsError:
                continue
            except OSError as exc:
                if exc.errno == errno.EEXIST:
                    continue
                if exc.errno == errno.EXDEV:
                    raise ValueError(
                        "Inbox and originals must be on the same filesystem"
                    ) from exc
                unsupported = {
                    errno.EPERM,
                    getattr(errno, "ENOTSUP", errno.EPERM),
                    getattr(errno, "EOPNOTSUPP", errno.EPERM),
                }
                if exc.errno in unsupported:
                    raise ValueError(
                        "Filesystem does not support required hard-link import"
                    ) from exc
                raise ValueError("Could not link complete import into originals") from exc
            return candidate

        raise AssertionError("candidate generation is infinite")

    @staticmethod
    def _candidate_name(source_name: str, index: int, name_max: int) -> str:
        extension = Path(source_name).suffix
        stem = source_name[: -len(extension)] if extension else source_name
        ordinal = "" if index == 1 else f"-{index}"
        try:
            fixed_size = len((ordinal + extension).encode("utf-8"))
            stem_budget = name_max - fixed_size
            if stem_budget < 0:
                raise ValueError
            encoded_stem = stem.encode("utf-8")
        except (UnicodeEncodeError, ValueError) as exc:
            raise ValueError(
                f"Source filename cannot fit originals filename limit: {source_name}"
            ) from exc
        if len(encoded_stem) > stem_budget:
            stem = encoded_stem[:stem_budget].decode("utf-8", errors="ignore")
        return f"{stem}{ordinal}{extension}"

    @staticmethod
    def _unlink_if_same(
        directory_fd: int,
        name: str,
        expected: os.stat_result,
    ) -> None:
        try:
            current = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        except OSError:
            return
        if (current.st_dev, current.st_ino) != (expected.st_dev, expected.st_ino):
            return
        try:
            os.unlink(name, dir_fd=directory_fd)
        except OSError:
            pass
