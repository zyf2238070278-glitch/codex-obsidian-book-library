import shutil
from pathlib import Path

from book_agent.config import AppPaths


class VaultManager:
    def __init__(self, paths: AppPaths) -> None:
        self.paths = paths

    def ensure_layout(self) -> None:
        directories = (
            self.paths.inbox,
            self.paths.originals,
            self.paths.parsed,
            self.paths.notes,
            self.paths.models,
            self.paths.database.parent,
        )
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

    def stage(self, source: Path) -> Path:
        resolved_source = source.resolve(strict=True)
        if not resolved_source.is_file():
            raise ValueError(f"Source is not a file: {source}")

        target = self._available_path(self.paths.inbox / resolved_source.name)
        return Path(shutil.copy2(resolved_source, target))

    def promote(self, staged: Path) -> Path:
        resolved_staged = staged.resolve(strict=True)
        resolved_staged.relative_to(self.paths.inbox.resolve(strict=True))

        target = self._available_path(self.paths.originals / resolved_staged.name)
        return resolved_staged.replace(target)

    @staticmethod
    def _available_path(path: Path) -> Path:
        if not path.exists():
            return path

        index = 2
        while True:
            candidate = path.with_name(f"{path.stem}-{index}{path.suffix}")
            if not candidate.exists():
                return candidate
            index += 1
