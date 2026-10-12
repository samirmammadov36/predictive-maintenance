# FD001 Training-Only Sensor Screening Report

This report is generated from `data/processed/sensor_screening.csv`, which is produced by `scripts/prepare_data.py` after the fixed 80/20 engine split.

Validation and official test rows are not used to decide which sensors are kept or removed.

## Frozen sensor list

**Selected (15):** `sensor_2`, `sensor_3`, `sensor_4`, `sensor_6`, `sensor_7`, `sensor_8`, `sensor_9`, `sensor_11`, `sensor_12`, `sensor_13`, `sensor_14`, `sensor_15`, `sensor_17`, `sensor_20`, `sensor_21`

**Removed (6):** `sensor_1`, `sensor_5`, `sensor_10`, `sensor_16`, `sensor_18`, `sensor_19`

## Screening evidence

| Sensor | Unique values | Unique ratio | Dominant fraction | Decision | Reason |
| --- | ---: | ---: | ---: | --- | --- |
| sensor_1 | 1 | 0.000060 | 1.000000 | Removed | constant |
| sensor_2 | 304 | 0.018118 | 0.009297 | Kept | kept |
| sensor_3 | 2922 | 0.174146 | 0.001311 | Kept | kept |
| sensor_4 | 3894 | 0.232076 | 0.001013 | Kept | kept |
| sensor_5 | 1 | 0.000060 | 1.000000 | Removed | constant |
| sensor_6 | 2 | 0.000119 | 0.980273 | Kept | kept |
| sensor_7 | 509 | 0.030336 | 0.005721 | Kept | kept |
| sensor_8 | 50 | 0.002980 | 0.057155 | Kept | kept |
| sensor_9 | 5731 | 0.341558 | 0.000894 | Kept | kept |
| sensor_10 | 1 | 0.000060 | 1.000000 | Removed | constant |
| sensor_11 | 158 | 0.009417 | 0.016747 | Kept | kept |
| sensor_12 | 423 | 0.025210 | 0.007211 | Kept | kept |
| sensor_13 | 54 | 0.003218 | 0.057870 | Kept | kept |
| sensor_14 | 5447 | 0.324632 | 0.000894 | Kept | kept |
| sensor_15 | 1878 | 0.111926 | 0.001907 | Kept | kept |
| sensor_16 | 1 | 0.000060 | 1.000000 | Removed | constant |
| sensor_17 | 13 | 0.000775 | 0.262590 | Kept | kept |
| sensor_18 | 1 | 0.000060 | 1.000000 | Removed | constant |
| sensor_19 | 1 | 0.000060 | 1.000000 | Removed | constant |
| sensor_20 | 117 | 0.006973 | 0.023780 | Kept | kept |
| sensor_21 | 4539 | 0.270517 | 0.001192 | Kept | kept |

## Rule

A sensor is removed if it is constant, or if both the configured near-constant unique-ratio and dominant-value criteria are satisfied.

The selected list is frozen before validation/test evaluation and is reused by the downstream preprocessing and model pipelines.
