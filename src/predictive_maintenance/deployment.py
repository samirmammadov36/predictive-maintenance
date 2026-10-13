"""Helpers for downloading deployment artifacts from GitHub Releases."""

from __future__ import annotations

from pathlib import Path
import tempfile
import zipfile

import requests


ARTIFACT_RELEASE_URL = (
    "https://github.com/samirmammadov36/"
    "predictive-maintenance/releases/download/"
    "deployment-v1/deployment_artifacts.zip"
)


def ensure_deployment_artifacts(root: str | Path) -> None:
    """Download deployment artifacts when they are not available locally."""
    root = Path(root)

    required_files = [
        root / "artifacts/models/baseline.joblib",
        root / "artifacts/models/lstm.pt",
        root / "artifacts/models/lstm_meta.joblib",
        root / "artifacts/models/random_forest.joblib",
        root / "artifacts/models/xgboost.json",
        root / "artifacts/preprocessing/prepared_bundle.joblib",
    ]

    if all(path.exists() for path in required_files):
        return

    artifacts_dir = root / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    temporary_path: Path | None = None

    try:
        with tempfile.NamedTemporaryFile(
            suffix=".zip",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)

            with requests.get(
                ARTIFACT_RELEASE_URL,
                stream=True,
                timeout=120,
            ) as response:
                response.raise_for_status()

                for chunk in response.iter_content(
                    chunk_size=1024 * 1024,
                ):
                    if chunk:
                        temporary_file.write(chunk)

        with zipfile.ZipFile(temporary_path) as archive:
            archive.extractall(artifacts_dir)

    except (
        requests.RequestException,
        zipfile.BadZipFile,
        OSError,
    ) as exc:
        raise RuntimeError(
            "Unable to download deployment artifacts."
        ) from exc

    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

    missing = [
        str(path)
        for path in required_files
        if not path.exists()
    ]

    if missing:
        raise RuntimeError(
            "Deployment artifacts were downloaded, "
            "but required files are missing: "
            + ", ".join(missing)
        )