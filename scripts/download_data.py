from __future__ import annotations

import argparse
import io
from pathlib import Path
import zipfile
import requests

NASA_ZIP_URL = "https://data.nasa.gov/docs/legacy/CMAPSSData.zip"
NEEDED = {"train_FD001.txt", "test_FD001.txt", "RUL_FD001.txt"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the official NASA C-MAPSS archive and extract FD001 files.")
    parser.add_argument("--out", default="data/raw", help="Destination directory")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"Downloading NASA C-MAPSS from: {NASA_ZIP_URL}")
    response = requests.get(NASA_ZIP_URL, timeout=120)
    response.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
        names = {Path(name).name: name for name in zf.namelist()}
        missing = NEEDED - names.keys()
        if missing:
            raise RuntimeError(f"NASA archive is missing expected files: {sorted(missing)}")
        for filename in sorted(NEEDED):
            data = zf.read(names[filename])
            path = out / filename
            path.write_bytes(data)
            print(f"Wrote {path} ({len(data):,} bytes)")

    print("FD001 download complete.")


if __name__ == "__main__":
    main()
