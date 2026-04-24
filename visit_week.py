import pandas as pd
import numpy as np

def get_output_schema():
    return pd.DataFrame({
        "客戶UUID": prep_string(),
        "業代": prep_string(),
        "拜訪紀錄UUID": prep_string(),
        "拜訪時間": prep_string(),
        # "拜訪時間_datetime": prep_datetime(),
        "拜訪日期": prep_date(),
        "拜訪週起始日": prep_date(),
        "檢查週": prep_date(),
        "是否有效拜訪": prep_int()
    })

def add_check_week(df):
    df = df.copy()

    # 1. 基本型別整理
    df["客戶UUID"] = df["客戶UUID"].astype(str)
    df["業代"] = df["業代"].astype(str)
    df["拜訪紀錄UUID"] = df["拜訪紀錄UUID"].astype(str)
    df["拜訪時間"] = df["拜訪時間"].astype(str)

    # 2. 清理拜訪時間字串
    # 例如: 2021-10-10T18:15:00Z[UTC]
    df["拜訪時間_清理後"] = (
        df["拜訪時間"]
        .str.replace(r"\[.*?\]", "", regex=True)
        .str.strip()
    )

    # 3. 轉成 UTC datetime
    拜訪時間_utc = pd.to_datetime(
        df["拜訪時間_清理後"],
        errors="coerce",
        utc=True
    )

    # 4. 轉成台灣時間
    拜訪時間_台灣 = 拜訪時間_utc.dt.tz_convert("Asia/Taipei")

    # 5. 拿掉時區
    拜訪時間_local = 拜訪時間_台灣.dt.tz_localize(None)

    df["拜訪時間_datetime_ts"] = 拜訪時間_local

    # 6. 拜訪日期
    df["拜訪日期_ts"] = pd.to_datetime(拜訪時間_local.dt.date, errors="coerce")

    # 7. 算拜訪週起始日（以週一為起點）
    df["拜訪週起始日_ts"] = (
        拜訪時間_local - pd.to_timedelta(拜訪時間_local.dt.weekday, unit="D")
    ).dt.normalize()

    # 8. 算檢查週（下一個週一）
    df["檢查週_ts"] = df["拜訪週起始日_ts"] + pd.Timedelta(days=7)

    # 9. 是否有效拜訪統一成 0/1
    df["是否有效拜訪"] = pd.to_numeric(df["是否有效拜訪"], errors="coerce").fillna(0)
    df["是否有效拜訪"] = np.where(df["是否有效拜訪"] >= 1, 1, 0).astype(int)

    # 10. Tableau Prep 需要合法格式字串
    df["拜訪時間_datetime"] = df["拜訪時間_datetime_ts"].dt.strftime("%Y-%m-%dT%H:%M:%S")
    df["拜訪日期"] = df["拜訪日期_ts"].dt.strftime("%Y-%m-%d")
    df["拜訪週起始日"] = df["拜訪週起始日_ts"].dt.strftime("%Y-%m-%d")
    df["檢查週"] = df["檢查週_ts"].dt.strftime("%Y-%m-%d")

    result = df[
        [
            "客戶UUID",
            "業代",
            "拜訪紀錄UUID",
            "拜訪時間",
            "拜訪時間_datetime",
            "拜訪日期",
            "拜訪週起始日",
            "檢查週",
            "是否有效拜訪"
        ]
    ].copy()

    return result