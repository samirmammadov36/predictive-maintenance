"""Deployment ZIP integrity, host-independent paths and transactional publication."""
from __future__ import annotations

import hashlib
from pathlib import Path
import stat
import sys
import zipfile

import pytest
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from predictive_maintenance import deployment


@pytest.fixture
def archive_factory(tmp_path, monkeypatch):
    def make(*, windows=False, extra=(), missing=(), pin=True):
        path = tmp_path / "download.zip"
        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
            def write(name, data):
                if isinstance(name, str):
                    entry = zipfile.ZipInfo(name)
                    # ZipInfo normalizes os.sep on Windows; retain genuine ZIP backslashes.
                    entry.filename = name
                    entry.orig_filename = name
                else:
                    entry = name
                archive.writestr(entry, data)
            for name in deployment.REQUIRED_FILES:
                if name not in missing:
                    write(name.replace("/", "\\") if windows else name, ("payload:" + name).encode())
            for name, data in extra:
                write(name, data)
        if pin:
            monkeypatch.setattr(deployment, "ARTIFACT_SHA256", hashlib.sha256(path.read_bytes()).hexdigest())
        return path
    return make


def assert_layout(destination):
    for name in deployment.REQUIRED_FILES:
        assert (destination / name).read_bytes() == ("payload:" + name).encode()
    assert not any("\\" in path.name for path in destination.rglob("*"))


@pytest.mark.parametrize("windows", [False, True])
def test_valid_checksum_and_portable_extraction_layout(archive_factory, tmp_path, windows):
    extra = [("models\\nested\\", b""), ("models\\nested\\note.txt", b"note")]
    path = archive_factory(windows=windows, extra=extra)
    with zipfile.ZipFile(path) as archive:
        if windows:
            assert "models\\random_forest.joblib" in [item.orig_filename for item in archive.infolist()]
    destination = tmp_path / "project" / "artifacts"
    deployment.extract_deployment_archive(path, destination)
    assert_layout(destination)
    assert (destination / "models/nested/note.txt").read_bytes() == b"note"
    assert not list(destination.parent.glob(".deployment-*"))


def test_checksum_mismatch_stops_before_extraction_and_preserves_existing(archive_factory, tmp_path):
    path = archive_factory()
    path.write_bytes(path.read_bytes() + b"tampered")
    destination = tmp_path / "artifacts"
    destination.mkdir()
    (destination / "local.txt").write_bytes(b"preserve")
    with pytest.raises(RuntimeError, match="SHA-256 mismatch: expected .* got"):
        deployment.extract_deployment_archive(path, destination)
    assert list(destination.iterdir()) == [destination / "local.txt"]
    assert (destination / "local.txt").read_bytes() == b"preserve"


@pytest.mark.parametrize("name", [
    "/escape", "\\escape", "C:\\escape", "C:escape", "//host/share/file",
    "\\\\host\\share\\file", "../escape", "models/../../escape", "models\\..\\escape",
    "models/../escape", "models/C:escape", ".", "models/file:stream", "models/bad\x00name",
])
def test_unsafe_paths_rejected_before_writing(archive_factory, tmp_path, name):
    path = archive_factory(extra=[(name, b"unsafe")])
    destination = tmp_path / "artifacts"
    with pytest.raises(RuntimeError, match="Unsafe deployment archive path"):
        deployment.extract_deployment_archive(path, destination)
    assert not destination.exists()
    assert not (tmp_path / "escape").exists()
    assert not list(tmp_path.glob(".deployment-*"))


@pytest.mark.parametrize("file_type", [stat.S_IFLNK, stat.S_IFIFO])
def test_symlink_and_special_entries_rejected(archive_factory, tmp_path, file_type):
    entry = zipfile.ZipInfo("models/link")
    entry.create_system = 3
    entry.external_attr = (file_type | 0o777) << 16
    path = archive_factory(extra=[(entry, b"../../outside")])
    with pytest.raises(RuntimeError, match="Symlinks/special files"):
        deployment.extract_deployment_archive(path, tmp_path / "artifacts")
    assert not (tmp_path / "artifacts").exists()


@pytest.mark.parametrize("extra", [
    [("models\\baseline.joblib", b"alias")],
    [("models//baseline.joblib", b"alias")],
    [("models/./baseline.joblib", b"alias")],
    [("models/BASELINE.joblib", b"alias")],
    [("Models/new.txt", b"alias")],
    [("models", b"not a directory")],
    [("models/item", b"file"), ("models/item/child", b"child")],
])
def test_normalized_path_and_file_directory_conflicts(archive_factory, tmp_path, extra):
    path = archive_factory(extra=extra)
    with pytest.raises(RuntimeError, match="Conflicting"):
        deployment.extract_deployment_archive(path, tmp_path / "artifacts")
    assert not (tmp_path / "artifacts").exists()


@pytest.mark.parametrize("missing", deployment.REQUIRED_FILES)
def test_each_required_file_must_be_in_archive_even_if_present_locally(archive_factory, tmp_path, missing):
    path = archive_factory(missing=[missing])
    destination = tmp_path / "artifacts"
    existing = destination / missing
    existing.parent.mkdir(parents=True)
    existing.write_bytes(b"existing")
    with pytest.raises(RuntimeError, match="missing required files"):
        deployment.extract_deployment_archive(path, destination)
    assert existing.read_bytes() == b"existing"
    assert [p for p in destination.rglob("*") if p.is_file()] == [existing]


def test_crc_failure_never_publishes_partial_artifacts(archive_factory, tmp_path, monkeypatch):
    path = archive_factory()
    payload = bytearray(path.read_bytes())
    offset = payload.index(b"payload:models/random_forest.joblib")
    payload[offset] ^= 1
    path.write_bytes(payload)
    monkeypatch.setattr(deployment, "ARTIFACT_SHA256", hashlib.sha256(payload).hexdigest())
    with pytest.raises(RuntimeError, match="CRC"):
        deployment.extract_deployment_archive(path, tmp_path / "artifacts")
    assert not (tmp_path / "artifacts").exists()
    assert not list(tmp_path.glob(".deployment-*"))


@pytest.mark.parametrize("existing", [False, True])
@pytest.mark.parametrize("failure", [OSError, RuntimeError])
def test_publish_failure_rolls_back_and_cleans_staging(archive_factory, tmp_path, monkeypatch, existing, failure):
    path = archive_factory()
    destination = tmp_path / "artifacts"
    if existing:
        destination.mkdir()
        (destination / "local.txt").write_bytes(b"preserve")
    rename = Path.rename
    def fail_publish(self, target):
        if self.name == "staged":
            raise failure("simulated publication failure")
        return rename(self, target)
    monkeypatch.setattr(Path, "rename", fail_publish)
    with pytest.raises(RuntimeError, match="simulated publication failure"):
        deployment.extract_deployment_archive(path, destination)
    if existing:
        assert (destination / "local.txt").read_bytes() == b"preserve"
        assert len(list(destination.iterdir())) == 1
    else:
        assert not destination.exists()
    assert not list(tmp_path.glob(".deployment-*"))


def test_success_preserves_unrelated_local_artifact_files(archive_factory, tmp_path):
    destination = tmp_path / "artifacts"
    (destination / "models").mkdir(parents=True)
    (destination / "models/.gitkeep").write_bytes(b"")
    (destination / "local.txt").write_bytes(b"preserve")
    (destination / "models/baseline.joblib").write_bytes(b"partial old")
    deployment.extract_deployment_archive(archive_factory(), destination)
    assert_layout(destination)
    assert (destination / "local.txt").read_bytes() == b"preserve"
    assert (destination / "models/.gitkeep").is_file()


class Response:
    def __init__(self, payload):
        self.payload = payload
    def __enter__(self):
        return self
    def __exit__(self, *args):
        return False
    def raise_for_status(self):
        pass
    def iter_content(self, chunk_size):
        yield self.payload[:10]
        yield b""
        yield self.payload[10:]


def test_downloader_validates_then_publishes_windows_archive(archive_factory, tmp_path, monkeypatch):
    payload = archive_factory(windows=True).read_bytes()
    calls = []
    def get(url, **kwargs):
        calls.append((url, kwargs))
        return Response(payload)
    monkeypatch.setattr(deployment.requests, "get", get)
    root = tmp_path / "project"
    deployment.ensure_deployment_artifacts(root)
    assert_layout(root / "artifacts")
    assert calls == [(deployment.ARTIFACT_RELEASE_URL, {"stream": True, "timeout": 120})]


def test_failed_download_checksum_cannot_be_mistaken_for_complete(archive_factory, tmp_path, monkeypatch):
    payload = archive_factory().read_bytes() + b"tampered"
    monkeypatch.setattr(deployment.requests, "get", lambda *a, **k: Response(payload))
    root = tmp_path / "project"
    for _ in range(2):
        with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
            deployment.ensure_deployment_artifacts(root)
        assert not (root / "artifacts").exists()


def test_network_failure_preserves_existing_incomplete_directory(tmp_path, monkeypatch):
    destination = tmp_path / "artifacts"
    destination.mkdir()
    (destination / "local.txt").write_bytes(b"preserve")
    def fail(*args, **kwargs):
        raise requests.HTTPError("503 unavailable")
    monkeypatch.setattr(deployment.requests, "get", fail)
    with pytest.raises(RuntimeError, match="503 unavailable"):
        deployment.ensure_deployment_artifacts(tmp_path)
    assert (destination / "local.txt").read_bytes() == b"preserve"
    assert len(list(destination.iterdir())) == 1


def test_complete_local_artifacts_skip_download_without_compatibility_claim(tmp_path, monkeypatch):
    for name in deployment.REQUIRED_FILES:
        path = tmp_path / "artifacts" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"local contents are not a compatibility proof")
    monkeypatch.setattr(deployment.requests, "get", lambda *a, **k: pytest.fail("Complete local layout downloaded"))
    deployment.ensure_deployment_artifacts(tmp_path)
    assert all((tmp_path / "artifacts" / name).read_bytes() == b"local contents are not a compatibility proof"
               for name in deployment.REQUIRED_FILES)


def test_invalid_zip_with_matching_checksum_is_not_published(tmp_path, monkeypatch):
    path = tmp_path / "bad.zip"
    path.write_bytes(b"not a ZIP")
    monkeypatch.setattr(deployment, "ARTIFACT_SHA256", hashlib.sha256(path.read_bytes()).hexdigest())
    with pytest.raises(RuntimeError, match="not a zip file"):
        deployment.extract_deployment_archive(path, tmp_path / "artifacts")
    assert not (tmp_path / "artifacts").exists()


def test_failed_rollback_keeps_original_artifacts_for_recovery(archive_factory, tmp_path, monkeypatch):
    path = archive_factory()
    destination = tmp_path / "artifacts"
    destination.mkdir()
    (destination / "local.txt").write_bytes(b"original")
    rename = Path.rename
    def fail(self, target):
        if self.name == "staged" or self.name.endswith("-previous"):
            raise OSError("simulated publication and rollback failure")
        return rename(self, target)
    monkeypatch.setattr(Path, "rename", fail)
    with pytest.raises(RuntimeError, match="original artifacts preserved at"):
        deployment.extract_deployment_archive(path, destination)
    assert not destination.exists()
    backups = list(tmp_path.glob(".deployment-*-previous"))
    assert len(backups) == 1 and (backups[0] / "local.txt").read_bytes() == b"original"
    assert list(tmp_path.glob(".deployment-*")) == backups
