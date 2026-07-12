import errno
import os
import shutil
import stat
from collections.abc import Iterator, Set
from contextlib import contextmanager
from pathlib import Path
from threading import Lock

from book_agent.config import AppPaths


_DIRECTORY_OPEN_FLAGS = (
    os.O_RDONLY
    | getattr(os, "O_CLOEXEC", 0)
    | getattr(os, "O_DIRECTORY", 0)
    | getattr(os, "O_NOFOLLOW", 0)
)
_SOURCE_OPEN_FLAGS = (
    os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
)
_RESERVE_OPEN_FLAGS = (
    os.O_CREAT
    | os.O_EXCL
    | os.O_WRONLY
    | getattr(os, "O_CLOEXEC", 0)
    | getattr(os, "O_NOFOLLOW", 0)
)
_FileIdentity = tuple[int, int]


class VaultManager:
    def __init__(self, paths: AppPaths) -> None:
        self.paths = paths
        self._ownership_lock = Lock()
        self._owned_staged: dict[Path, _FileIdentity] = {}
        self._issued_names: set[str] = set()
        self._reserved_names: set[str] = set()

    def ensure_layout(self) -> None:
        directories = (
            ("inbox", self.paths.inbox),
            ("originals", self.paths.originals),
            ("parsed", self.paths.parsed),
            ("notes", self.paths.notes),
            ("models", self.paths.models),
            ("database.parent", self.paths.database.parent),
        )
        for label, directory in directories:
            with self._managed_directory(directory, label, create=True):
                pass

    def stage(self, source: Path) -> Path:
        resolved_source, source_fd, source_info = self._open_source(source)
        try:
            with self._managed_directory(
                self.paths.inbox,
                "inbox",
                create=False,
            ) as (inbox, inbox_fd):
                target_name, target_fd, target_info = self._reserve_stage_target(
                    inbox_fd,
                    resolved_source.name,
                    stat.S_IMODE(source_info.st_mode),
                )
                try:
                    self._copy_to_reserved_target(
                        source_fd,
                        source_info,
                        inbox_fd,
                        target_name,
                        target_fd,
                        target_info,
                        source,
                    )
                except BaseException:
                    self._release_stage_reservation(target_name)
                    raise

                staged = inbox / target_name
                self._record_staged_ownership(staged, target_name, target_info)
                return staged
        finally:
            os.close(source_fd)

    def promote(self, staged: Path) -> Path:
        staged_path, identity = self._consume_staged_ownership(staged)
        try:
            return self._promote_owned(staged_path, identity)
        except BaseException:
            self._restore_staged_ownership(staged_path, identity)
            raise

    def _promote_owned(
        self,
        staged: Path,
        identity: _FileIdentity,
    ) -> Path:
        with self._managed_directory(
            self.paths.inbox,
            "inbox",
            create=False,
        ) as (inbox, inbox_fd):
            staged_name, staged_info = self._direct_regular_entry(
                staged,
                inbox,
                inbox_fd,
            )
            if self._identity(staged_info) != identity:
                raise ValueError(
                    f"Staged file no longer matches the generation created by this "
                    f"VaultManager: {staged}"
                )
            with self._managed_directory(
                self.paths.originals,
                "originals",
                create=False,
            ) as (originals, originals_fd):
                claim_name = self._claim_staged_entry(
                    inbox_fd,
                    staged_name,
                    staged_info,
                    staged,
                )
                try:
                    for target_name in self._candidate_names(staged_name):
                        try:
                            os.link(
                                claim_name,
                                target_name,
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
                            raise ValueError(
                                f"Could not safely promote staged file: {staged}"
                            ) from exc

                        cleanup_info = staged_info
                        try:
                            current_claim_info = os.stat(
                                claim_name,
                                dir_fd=inbox_fd,
                                follow_symlinks=False,
                            )
                            linked_info = os.stat(
                                target_name,
                                dir_fd=originals_fd,
                                follow_symlinks=False,
                            )
                            if self._same_file(current_claim_info, linked_info):
                                cleanup_info = linked_info
                            if not self._same_file(
                                current_claim_info,
                                staged_info,
                            ) or not self._same_file(linked_info, staged_info):
                                raise ValueError(
                                    f"Staged file changed during promotion: {staged}"
                                )
                            os.unlink(claim_name, dir_fd=inbox_fd)
                        except ValueError:
                            self._unlink_if_same(
                                originals_fd,
                                target_name,
                                cleanup_info,
                            )
                            raise
                        except OSError as exc:
                            self._unlink_if_same(
                                originals_fd,
                                target_name,
                                cleanup_info,
                            )
                            raise ValueError(
                                f"Could not safely complete promotion: {staged}"
                            ) from exc

                        return originals / target_name
                except BaseException:
                    self._restore_claim(inbox_fd, claim_name, staged_name)
                    raise

        raise AssertionError("candidate name generation is infinite")

    def _reserve_stage_target(
        self,
        directory_fd: int,
        filename: str,
        mode: int,
    ) -> tuple[str, int, os.stat_result]:
        with self._ownership_lock:
            unavailable_names = self._issued_names | self._reserved_names
            reserved = self._reserve_target(
                directory_fd,
                filename,
                mode,
                excluded_names=unavailable_names,
            )
            self._reserved_names.add(reserved[0])
            return reserved

    def _record_staged_ownership(
        self,
        staged: Path,
        staged_name: str,
        staged_info: os.stat_result,
    ) -> None:
        with self._ownership_lock:
            self._reserved_names.discard(staged_name)
            self._issued_names.add(staged_name)
            self._owned_staged[staged] = self._identity(staged_info)

    def _release_stage_reservation(self, staged_name: str) -> None:
        with self._ownership_lock:
            self._reserved_names.discard(staged_name)

    def _consume_staged_ownership(
        self,
        staged: Path,
    ) -> tuple[Path, _FileIdentity]:
        staged_path = self._normalized_staged_path(staged)
        with self._ownership_lock:
            try:
                identity = self._owned_staged.pop(staged_path)
            except KeyError as exc:
                raise ValueError(
                    f"Staged path was not created by this VaultManager or has "
                    f"already been promoted: {staged}"
                ) from exc
        return staged_path, identity

    def _restore_staged_ownership(
        self,
        staged: Path,
        identity: _FileIdentity,
    ) -> None:
        try:
            current = staged.lstat()
        except OSError:
            return
        if not stat.S_ISREG(current.st_mode) or self._identity(current) != identity:
            return

        with self._ownership_lock:
            self._owned_staged.setdefault(staged, identity)

    @staticmethod
    def _normalized_staged_path(staged: Path) -> Path:
        try:
            expanded = Path(staged).expanduser()
            return Path(os.path.abspath(os.fspath(expanded)))
        except (OSError, RuntimeError, TypeError) as exc:
            raise ValueError(
                f"Staged path was not created by this VaultManager: {staged}"
            ) from exc

    def _open_source(self, source: Path) -> tuple[Path, int, os.stat_result]:
        try:
            expanded_source = Path(source).expanduser()
            resolved_source = expanded_source.resolve(strict=True)
        except (OSError, RuntimeError, TypeError) as exc:
            raise ValueError(f"Source file is unavailable: {source}") from exc

        try:
            source_fd = os.open(resolved_source, _SOURCE_OPEN_FLAGS)
        except OSError as exc:
            raise ValueError(f"Source file cannot be opened safely: {source}") from exc

        try:
            source_info = os.fstat(source_fd)
        except OSError as exc:
            os.close(source_fd)
            raise ValueError(
                f"Source file cannot be inspected safely: {source}"
            ) from exc

        if not stat.S_ISREG(source_info.st_mode):
            os.close(source_fd)
            raise ValueError(f"Source must be a regular file: {source}")

        return resolved_source, source_fd, source_info

    @contextmanager
    def _managed_directory(
        self,
        configured: Path,
        label: str,
        *,
        create: bool,
    ) -> Iterator[tuple[Path, int]]:
        root = self._project_root(create=create)
        directory = self._confined_path(configured, label, root)
        components = directory.relative_to(root).parts

        try:
            current_fd = os.open(root, _DIRECTORY_OPEN_FLAGS)
        except OSError as exc:
            raise ValueError(f"Project root cannot be opened safely: {root}") from exc

        current_path = root
        try:
            for component in components:
                if create:
                    try:
                        os.mkdir(component, dir_fd=current_fd)
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
                        f"Managed directory '{label}' is unavailable: "
                        f"{current_path / component}"
                    ) from exc

                if stat.S_ISLNK(entry_info.st_mode):
                    raise ValueError(
                        f"Managed directory '{label}' contains a symlink component: "
                        f"{current_path / component}"
                    )
                if not stat.S_ISDIR(entry_info.st_mode):
                    raise ValueError(
                        f"Managed directory '{label}' contains a non-directory "
                        f"component: {current_path / component}"
                    )

                try:
                    next_fd = os.open(
                        component,
                        _DIRECTORY_OPEN_FLAGS,
                        dir_fd=current_fd,
                    )
                except OSError as exc:
                    raise ValueError(
                        f"Managed directory '{label}' cannot be opened safely: "
                        f"{current_path / component}"
                    ) from exc

                try:
                    opened_info = os.fstat(next_fd)
                except OSError as exc:
                    os.close(next_fd)
                    raise ValueError(
                        f"Managed directory '{label}' cannot be inspected safely: "
                        f"{current_path / component}"
                    ) from exc
                if not self._same_file(entry_info, opened_info):
                    os.close(next_fd)
                    raise ValueError(
                        f"Managed directory '{label}' changed during validation: "
                        f"{current_path / component}"
                    )

                previous_fd = current_fd
                current_fd = next_fd
                os.close(previous_fd)
                current_path /= component

            yield directory, current_fd
        finally:
            os.close(current_fd)

    def _project_root(self, *, create: bool) -> Path:
        try:
            expanded_root = Path(self.paths.root).expanduser()
            root = Path(os.path.abspath(os.fspath(expanded_root)))
            unresolved_root = root.resolve(strict=False)
        except (OSError, RuntimeError, TypeError) as exc:
            raise ValueError(f"Project root path is invalid: {self.paths.root}") from exc

        if unresolved_root != root:
            raise ValueError(f"Project root contains a symlink component: {root}")

        if create:
            try:
                root.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                raise ValueError(f"Project root cannot be created: {root}") from exc

        try:
            root_info = root.lstat()
        except OSError as exc:
            raise ValueError(f"Project root is unavailable: {root}") from exc

        if stat.S_ISLNK(root_info.st_mode) or not stat.S_ISDIR(root_info.st_mode):
            raise ValueError(f"Project root must be a real directory: {root}")

        try:
            resolved_root = root.resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise ValueError(f"Project root cannot be resolved safely: {root}") from exc
        if resolved_root != root:
            raise ValueError(f"Project root contains a symlink component: {root}")

        return root

    @staticmethod
    def _confined_path(configured: Path, label: str, root: Path) -> Path:
        try:
            expanded = Path(configured).expanduser()
            if not expanded.is_absolute():
                expanded = root / expanded
            directory = Path(os.path.abspath(os.fspath(expanded)))
            directory.relative_to(root)
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            raise ValueError(
                f"Managed directory '{label}' is not confined beneath project root: "
                f"{configured}"
            ) from exc
        return directory

    @staticmethod
    def _reserve_target(
        directory_fd: int,
        filename: str,
        mode: int,
        *,
        excluded_names: Set[str] = frozenset(),
    ) -> tuple[str, int, os.stat_result]:
        for candidate_name in VaultManager._candidate_names(filename):
            if candidate_name in excluded_names:
                continue
            try:
                target_fd = os.open(
                    candidate_name,
                    _RESERVE_OPEN_FLAGS,
                    mode,
                    dir_fd=directory_fd,
                )
            except FileExistsError:
                continue
            except OSError as exc:
                if exc.errno == errno.EEXIST:
                    continue
                raise ValueError(
                    f"Could not reserve a safe inbox filename for: {filename}"
                ) from exc
            try:
                target_info = os.fstat(target_fd)
            except OSError as exc:
                try:
                    os.close(target_fd)
                except OSError:
                    pass
                try:
                    os.unlink(candidate_name, dir_fd=directory_fd)
                except OSError:
                    pass
                raise ValueError(
                    f"Could not inspect reserved inbox target: {filename}"
                ) from exc
            return candidate_name, target_fd, target_info

        raise AssertionError("candidate name generation is infinite")

    @staticmethod
    def _copy_to_reserved_target(
        source_fd: int,
        source_info: os.stat_result,
        directory_fd: int,
        target_name: str,
        target_fd: int,
        reserved_info: os.stat_result,
        source: Path,
    ) -> None:
        completed = False
        try:
            with os.fdopen(os.dup(source_fd), "rb") as source_stream:
                with os.fdopen(os.dup(target_fd), "wb") as target_stream:
                    shutil.copyfileobj(source_stream, target_stream)
                    target_stream.flush()

            os.fchmod(target_fd, stat.S_IMODE(source_info.st_mode))
            os.utime(
                target_fd,
                ns=(source_info.st_atime_ns, source_info.st_mtime_ns),
            )
            if not VaultManager._entry_matches(
                directory_fd,
                target_name,
                reserved_info,
            ):
                raise ValueError(
                    f"Reserved inbox target changed while staging: {target_name}"
                )
            completed = True
        except ValueError:
            raise
        except Exception as exc:
            raise ValueError(f"Could not safely stage source file: {source}") from exc
        finally:
            try:
                os.close(target_fd)
            finally:
                if not completed:
                    VaultManager._unlink_if_same(
                        directory_fd,
                        target_name,
                        reserved_info,
                    )

    @staticmethod
    def _direct_regular_entry(
        staged: Path,
        inbox: Path,
        inbox_fd: int,
    ) -> tuple[str, os.stat_result]:
        try:
            expanded = Path(staged).expanduser()
            staged_path = Path(os.path.abspath(os.fspath(expanded)))
        except (OSError, RuntimeError, TypeError) as exc:
            raise ValueError(
                f"Staged path must name a direct regular file in inbox: {staged}"
            ) from exc

        if staged_path.parent != inbox:
            raise ValueError(
                f"Staged path must name a direct regular file in inbox: {staged}"
            )

        try:
            staged_info = os.stat(
                staged_path.name,
                dir_fd=inbox_fd,
                follow_symlinks=False,
            )
        except OSError as exc:
            raise ValueError(
                f"Staged path must name a direct regular file in inbox: {staged}"
            ) from exc

        if not stat.S_ISREG(staged_info.st_mode):
            raise ValueError(
                f"Staged path must name a direct regular file in inbox: {staged}"
            )

        return staged_path.name, staged_info

    @staticmethod
    def _claim_staged_entry(
        inbox_fd: int,
        staged_name: str,
        staged_info: os.stat_result,
        staged: Path,
    ) -> str:
        claim_name, claim_fd, placeholder_info = VaultManager._reserve_target(
            inbox_fd,
            f".{staged_name}.claim",
            0o600,
        )
        try:
            os.close(claim_fd)
        except OSError as exc:
            VaultManager._unlink_if_same(inbox_fd, claim_name, placeholder_info)
            raise ValueError(f"Could not prepare staged file claim: {staged}") from exc

        try:
            os.replace(
                staged_name,
                claim_name,
                src_dir_fd=inbox_fd,
                dst_dir_fd=inbox_fd,
            )
        except FileNotFoundError as exc:
            VaultManager._unlink_if_same(inbox_fd, claim_name, placeholder_info)
            raise ValueError(
                f"Staged file is already being promoted or unavailable: {staged}"
            ) from exc
        except OSError as exc:
            VaultManager._unlink_if_same(inbox_fd, claim_name, placeholder_info)
            raise ValueError(f"Could not safely claim staged file: {staged}") from exc

        try:
            claimed_info = os.stat(
                claim_name,
                dir_fd=inbox_fd,
                follow_symlinks=False,
            )
        except OSError as exc:
            VaultManager._restore_claim(inbox_fd, claim_name, staged_name)
            raise ValueError(f"Could not inspect staged file claim: {staged}") from exc
        if not VaultManager._same_file(claimed_info, staged_info):
            VaultManager._restore_claim(inbox_fd, claim_name, staged_name)
            raise ValueError(f"Staged file changed while being claimed: {staged}")

        return claim_name

    @staticmethod
    def _restore_claim(inbox_fd: int, claim_name: str, staged_name: str) -> None:
        try:
            os.link(
                claim_name,
                staged_name,
                src_dir_fd=inbox_fd,
                dst_dir_fd=inbox_fd,
                follow_symlinks=False,
            )
        except OSError:
            return
        try:
            os.unlink(claim_name, dir_fd=inbox_fd)
        except OSError:
            pass

    @staticmethod
    def _candidate_names(filename: str) -> Iterator[str]:
        yield filename

        path = Path(filename)
        index = 2
        while True:
            yield f"{path.stem}-{index}{path.suffix}"
            index += 1

    @staticmethod
    def _same_file(first: os.stat_result, second: os.stat_result) -> bool:
        return VaultManager._identity(first) == VaultManager._identity(second)

    @staticmethod
    def _identity(file_info: os.stat_result) -> _FileIdentity:
        return file_info.st_dev, file_info.st_ino

    @staticmethod
    def _entry_matches(
        directory_fd: int,
        name: str,
        expected: os.stat_result,
    ) -> bool:
        try:
            current = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        except OSError:
            return False
        return VaultManager._same_file(current, expected)

    @staticmethod
    def _unlink_if_same(
        directory_fd: int,
        name: str,
        expected: os.stat_result,
    ) -> None:
        if not VaultManager._entry_matches(directory_fd, name, expected):
            return
        try:
            os.unlink(name, dir_fd=directory_fd)
        except OSError:
            pass
