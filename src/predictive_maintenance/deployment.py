"""Verified, staged deployment-artifact loading from GitHub Releases."""

from __future__ import annotations

import hashlib
import ntpath
from pathlib import Path, PurePosixPath
import shutil
import stat
import tempfile
import zipfile

import requests


ARTIFACT_RELEASE_URL = (
    "https://github.com/samirmammadov36/"
    "predictive-maintenance/releases/download/"
    "deployment-v1/deployment_artifacts.zip"
)
ARTIFACT_SHA256 = "fb1311d9e86f90efe3f2b64e71a75193c7ba1fcc0d8f502e9dfb271985f3ddb2"
REQUIRED_FILES = (
    "models/baseline.joblib",
    "models/lstm.pt",
    "models/lstm_meta.joblib",
    "models/random_forest.joblib",
    "models/xgboost.json",
    "preprocessing/prepared_bundle.joblib",
)


def _archive_entries(archive: zipfile.ZipFile):
    """Normalize ZIP paths independently of the host OS before any extraction."""
    entries = {}
    spellings = {}
    for info in archive.infolist():
        name = info.orig_filename.replace("\\", "/")
        if (not name or "\x00" in name or name.startswith("/")
                or ntpath.splitdrive(name)[0]
                or any(part == ".." or ":" in part for part in name.split("/"))):
            raise ValueError(f"Unsafe deployment archive path: {info.orig_filename!r}")
        path = PurePosixPath(name)
        if not path.parts:
            raise ValueError(f"Unsafe deployment archive path: {name!r}")
        mode = stat.S_IFMT(info.external_attr >> 16)
        if mode not in (0, stat.S_IFREG, stat.S_IFDIR):
            raise ValueError(f"Symlinks/special files are forbidden in deployment ZIP: {name!r}")
        directory = name.endswith("/") or mode == stat.S_IFDIR
        normalized = path.as_posix()
        if normalized in entries:
            raise ValueError(f"Conflicting normalized archive path: {normalized}")
        # Reject case aliases too: the same archive must be unambiguous on both OSes.
        for length in range(1, len(path.parts) + 1):
            prefix = PurePosixPath(*path.parts[:length]).as_posix()
            previous = spellings.setdefault(prefix.casefold(), prefix)
            if previous != prefix:
                raise ValueError(f"Conflicting normalized archive paths: {previous}, {prefix}")
        entries[normalized] = (info, path, directory)
    for normalized, (_, path, _) in entries.items():
        for parent in path.parents:
            if parent.as_posix() in entries and not entries[parent.as_posix()][2]:
                raise ValueError(f"Conflicting file/directory archive paths: {normalized}")
    missing = [name for name in REQUIRED_FILES if name not in entries or entries[name][2]]
    if missing:
        raise ValueError("Deployment archive is missing required files: " + ", ".join(missing))
    return list(entries.values())


def extract_deployment_archive(archive_path: str | Path, artifacts_dir: str | Path) -> None:
    """Verify and stage a ZIP, then replace the directory with rollback on failure.

    Required-file checks establish layout only, not model/preprocessing compatibility.
    Existing unrelated local artifact files are retained; complete local deployments
    bypass this helper through ensure_deployment_artifacts().
    """
    destination = Path(artifacts_dir)
    try:
        with Path(archive_path).open("rb") as source:
            checksum = hashlib.file_digest(source, "sha256").hexdigest()
            if checksum != ARTIFACT_SHA256:
                raise ValueError(
                    f"Deployment archive SHA-256 mismatch: expected {ARTIFACT_SHA256}, got {checksum}"
                )
            source.seek(0)
            with zipfile.ZipFile(source) as archive:
                entries = _archive_entries(archive)
                if destination.is_symlink() or (destination.exists() and not destination.is_dir()):
                    raise ValueError("Artifact destination must be a directory, not a symlink/file")
                destination.parent.mkdir(parents=True, exist_ok=True)
                with tempfile.TemporaryDirectory(prefix=".deployment-", dir=destination.parent) as temporary:
                    staging = Path(temporary) / "staged"
                    staging.mkdir()
                    if destination.exists():
                        if any(path.is_symlink() for path in destination.rglob("*")):
                            raise ValueError("Existing incomplete artifact directory contains a symlink")
                        shutil.copytree(destination, staging, dirs_exist_ok=True)
                    for info, relative, directory in entries:
                        target = staging.joinpath(*relative.parts)
                        if not target.resolve().is_relative_to(staging.resolve()):
                            raise ValueError(f"Archive entry escapes artifact staging: {relative}")
                        if directory:
                            target.mkdir(parents=True, exist_ok=True)
                        else:
                            target.parent.mkdir(parents=True, exist_ok=True)
                            with archive.open(info) as payload, target.open("wb") as output:
                                shutil.copyfileobj(payload, output)
                    missing = [name for name in REQUIRED_FILES if not (staging / name).is_file()]
                    if missing:
                        raise ValueError("Staged deployment is missing required files: " + ", ".join(missing))
                    # Keep the backup outside auto-cleaned staging if rollback itself fails.
                    backup = Path(temporary).with_name(Path(temporary).name + "-previous")
                    had_destination = destination.exists()
                    if had_destination:
                        destination.rename(backup)
                    try:
                        staging.rename(destination)
                    except BaseException:
                        if had_destination:
                            try:
                                backup.rename(destination)
                            except OSError as exc:
                                raise RuntimeError(
                                    f"Publication/rollback failed; original artifacts preserved at {backup}"
                                ) from exc
                        raise
                    if had_destination:
                        shutil.rmtree(backup)
    except (OSError, ValueError, zipfile.BadZipFile, RuntimeError) as exc:
        raise RuntimeError(f"Unable to extract deployment artifacts: {exc}") from exc


def ensure_deployment_artifacts(root: str | Path) -> None:
    """Reuse a complete local layout; otherwise download and verify the pinned ZIP.

    Local file existence does not prove model compatibility. The existing inference
    loaders remain responsible for their model/preprocessing checks.
    """
    artifacts_dir = Path(root) / "artifacts"
    if all((artifacts_dir / name).is_file() for name in REQUIRED_FILES):
        return
    try:
        with tempfile.TemporaryDirectory(prefix="deployment-download-") as temporary:
            archive_path = Path(temporary) / "deployment_artifacts.zip"
            with archive_path.open("wb") as output:
                with requests.get(ARTIFACT_RELEASE_URL, stream=True, timeout=120) as response:
                    response.raise_for_status()
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            output.write(chunk)
            extract_deployment_archive(archive_path, artifacts_dir)
    except (requests.RequestException, OSError) as exc:
        raise RuntimeError(f"Unable to download deployment artifacts: {exc}") from exc
