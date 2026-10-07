ID_COLUMN = "engine_id"
CYCLE_COLUMN = "cycle"
SETTING_COLUMNS = [f"setting_{i}" for i in range(1, 4)]
SENSOR_COLUMNS = [f"sensor_{i}" for i in range(1, 22)]
RAW_COLUMNS = [ID_COLUMN, CYCLE_COLUMN, *SETTING_COLUMNS, *SENSOR_COLUMNS]
RUL_TRUTH_COLUMN = "rul_true"
RUL_TARGET_COLUMN = "rul_target"
