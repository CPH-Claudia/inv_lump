import pandas as pd
import numpy as np
from datetime import datetime, timedelta

VISIT_ID = "拜訪紀錄UUID"
VISIT_DT = "拜訪時間"
NOTE_COL = "拜訪備註"

def get_output_schema():
    return pd.DataFrame({
        VISIT_ID: prep_string(),
        "是否在上一週": prep_string(),
        "是否含#躉投": prep_string(),
        "是否有效拜訪": prep_string(),
        "拜訪時間_解析後": prep_string(),
    }, index=[0])

def parse_dt_series(s: pd.Series) -> pd.Series:
    # 先轉字串
    x = s.fillna("").astype(str).str.strip()

    # 去掉尾巴的 [UTC]
    x = x.str.replace("[UTC]", "", regex=False)

    # 若有 Z 保留沒關係，pandas 可解析
    dt = pd.to_datetime(x, errors="coerce", utc=True)

    # 轉成 naive datetime，方便後面跟本地時間比較
    dt = dt.dt.tz_convert(None)

    return dt

def execute(df):
    data = df.copy()
    data.columns = [str(c).strip() for c in data.columns]

    required_cols = [VISIT_ID, VISIT_DT, NOTE_COL]
    missing_cols = [c for c in required_cols if c not in data.columns]
    if missing_cols:
        raise ValueError("缺少必要欄位: " + str(missing_cols))

    data = data[[VISIT_ID, VISIT_DT, NOTE_COL]].copy()

    data[VISIT_ID] = data[VISIT_ID].fillna("").astype(str)
    data[NOTE_COL] = data[NOTE_COL].fillna("").astype(str)

    visit_dt = parse_dt_series(data[VISIT_DT])

    # 本週週一
    today = datetime.today()
    this_monday = today - timedelta(days=today.weekday())

    # 上週：上週一 ~ 本週一
    last_week_start = this_monday - timedelta(days=7)
    last_week_end = this_monday

    is_last_week = (visit_dt >= last_week_start) & (visit_dt < last_week_end)
    has_tag = data[NOTE_COL].str.contains("#車險", na=False)

    result = pd.DataFrame({
        VISIT_ID: data[VISIT_ID].astype(str),
        "是否在上一週": np.where(is_last_week, "1", "0"),
        "是否含#躉投": np.where(has_tag, "1", "0"),
        "是否有效拜訪": np.where(is_last_week & has_tag, "1", "0"),
        "拜訪時間_解析後": visit_dt.dt.strftime("%Y-%m-%d %H:%M:%S").fillna(""),
    })

    for c in result.columns:
        result[c] = result[c].fillna("").astype(str)

    return result
