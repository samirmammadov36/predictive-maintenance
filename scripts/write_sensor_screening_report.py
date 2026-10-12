"""Write a committed Markdown summary of the training-only sensor screening result."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data/processed/sensor_screening.csv"
OUTPUT = ROOT / "docs/sensor_screening_report.md"

REQUIRED_COLUMNS = {
    "sensor",
    "nunique",
    "unique_ratio",
    "dominant_fraction",
    "removed",
    "reason",
}


def render_report(screening: pd.DataFrame) -> str:
    missing = sorted(REQUIRED_COLUMNS - set(screening.columns))
    if missing:
        raise ValueError(f"Sensor screening CSV is missing columns: {missing}")

    ordered = screening.copy()
    ordered["removed"] = ordered["removed"].astype(bool)
    selected = ordered.loc[~ordered["removed"], "sensor"].tolist()
    removed = ordered.loc[ordered["removed"], "sensor"].tolist()

    lines = [
        "# FD001 Training-Only Sensor Screening Report",
        "",
        "This report is generated from `data/processed/sensor_screening.csv`, which is produced by `scripts/prepare_data.py` after the fixed 80/20 engine split.",
        "",
        "Validation and official test rows are not used to decide which sensors are kept or removed.",
        "",
        "## Frozen sensor list",
        "",
        f"**Selected ({len(selected)}):** " + (", ".join(f"`{s}`" for s in selected) if selected else "None"),
        "",
        f"**Removed ({len(removed)}):** " + (", ".join(f"`{s}`" for s in removed) if removed else "None"),
        "",
        "## Screening evidence",
        "",
        "| Sensor | Unique values | Unique ratio | Dominant fraction | Decision | Reason |",
        "| --- | ---: | ---: | ---: | --- | --- |",
    ]

    for row in ordered.itertuples(index=False):
        decision = "Removed" if bool(row.removed) else "Kept"
        lines.append(
            f"| {row.sensor} | {int(row.nunique)} | {float(row.unique_ratio):.6f} | "
            f"{float(row.dominant_fraction):.6f} | {decision} | {row.reason} |"
        )

    lines.extend(
        [
            "",
            "## Rule",
            "",
            "A sensor is removed if it is constant, or if both the configured near-constant unique-ratio and dominant-value criteria are satisfied.",
            "",
            "The selected list is frozen before validation/test evaluation and is reused by the downstream preprocessing and model pipelines.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    if not INPUT.exists():
        raise FileNotFoundError(
            f"Missing {INPUT}. Run `python scripts/prepare_data.py` first so the training-only screening is generated."
        )

    screening = pd.read_csv(INPUT)
    OUTPUT.write_text(render_report(screening), encoding="utf-8")
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
