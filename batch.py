import pandas as pd
import numpy as np

# ===== 欄位名稱 =====
AGENT_COL = "業代"
CUST_COL = "客戶識別碼"
NAME_COL = "客戶姓名"
DISTRICT_COL = "通訊地行政區_整合"
ZIP_COL = "通訊地郵遞區號_整合"
SOURCE_COL = "名單來源類型"

SLEEP_FLAG = "是否靜止戶"
LUMP_FLAG = "是否躉投"

VISIT_ID = "拜訪紀錄UUID"
VISIT_DT = "拜訪時間"
NOTE_COL = "拜訪備註"

RELEASE_DATE = pd.Timestamp("2026-03-15")
BATCH_SIZE = 20
UNLOCK_THRESHOLD = 1


# ===== Schema =====
def get_output_schema():
    return pd.DataFrame({
        AGENT_COL: prep_string(),
        CUST_COL: prep_string(),
        NAME_COL: prep_string(),
        DISTRICT_COL: prep_string(),
        ZIP_COL: prep_string(),
        "姓氏": prep_string(),

        SLEEP_FLAG: prep_int(),
        LUMP_FLAG: prep_int(),
        SOURCE_COL: prep_string(),
        "是否雙重命中": prep_int(),

        "名單排序順位": prep_int(),
        # "批次": prep_int(),
        "總批次": prep_int(),

        "躉投排序順位": prep_int(),
        "躉投批次": prep_int(),
        "靜止戶排序順位": prep_int(),
        "靜止戶批次": prep_int(),

        "目前開放躉投批次": prep_int(),
        "目前開放靜止戶批次": prep_int(),

        "躉投批次顯示": prep_string(),
        "靜止戶批次顯示": prep_string(),

        "是否釋出": prep_int(),

        "有效拜訪客戶數": prep_int(),
        "躉投有效拜訪客戶數": prep_int(),
        "靜止戶有效拜訪客戶數": prep_int(),
        "距離解鎖下一批": prep_int(),

        "是否有效拜訪": prep_int(),
        "是否含#躉投": prep_int(),
        "是否含#靜止": prep_int(),

        "有效拜訪時間": prep_string(),

        "郵遞區號_排序用": prep_int(),
        "地區排序輔助": prep_string(),

    }, index=[0])


# ===== 時間解析 =====
def parse_dt_series(s):
    x = s.fillna("").astype(str).str.strip()
    x = x.str.replace("[UTC]", "", regex=False)
    dt = pd.to_datetime(x, errors="coerce", utc=True)
    return dt.dt.tz_convert(None)


def to_int_flag(s):
    return pd.to_numeric(s, errors="coerce").fillna(0).astype(int)


# ===== 主程式 =====
def execute(df):

    data = df.copy()
    data.columns = [str(c).strip() for c in data.columns]

    required_cols = [
        AGENT_COL, CUST_COL, NAME_COL, DISTRICT_COL, ZIP_COL,
        SLEEP_FLAG, LUMP_FLAG, SOURCE_COL,
        VISIT_ID, VISIT_DT, NOTE_COL
    ]

    for c in required_cols:
        if c not in data.columns:
            raise ValueError(f"缺少欄位: {c}")

    # ===== 清理 =====
    for c in [AGENT_COL, CUST_COL, NAME_COL, DISTRICT_COL, ZIP_COL, VISIT_ID, NOTE_COL]:
        data[c] = data[c].fillna("").astype(str)

    data[SLEEP_FLAG] = to_int_flag(data[SLEEP_FLAG])
    data[LUMP_FLAG] = to_int_flag(data[LUMP_FLAG])

    data["拜訪時間_解析"] = parse_dt_series(data[VISIT_DT])

    note = data[NOTE_COL].fillna("")

    # ===== 標籤 =====
    data["是否名單釋出日後"] = data["拜訪時間_解析"] >= RELEASE_DATE

    data["是否含#躉投_row"] = np.where(
        data["是否名單釋出日後"] & note.str.contains("#2026夏賽"),
        1, 0
    )

    data["是否含#靜止_row"] = np.where(
        data["是否名單釋出日後"] & note.str.contains("#靜止"),
        1, 0
    )

    data["是否有效拜訪_row"] = np.where(
        (data["是否含#躉投_row"] == 1) | (data["是否含#靜止_row"] == 1),
        1, 0
    )

    data["有效拜訪時間_row"] = data["拜訪時間_解析"].where(data["是否有效拜訪_row"] == 1)

    # ===== 客戶彙總 =====
    customer = (
        data.groupby([AGENT_COL, CUST_COL])
        .agg({
            NAME_COL: "first",
            DISTRICT_COL: "first",
            ZIP_COL: "first",
            SLEEP_FLAG: "max",
            LUMP_FLAG: "max",
            SOURCE_COL: "first",
            "是否有效拜訪_row": "max",
            "是否含#躉投_row": "max",
            "是否含#靜止_row": "max",
            "有效拜訪時間_row": "min"
        })
        .reset_index()
    )

    customer["姓氏"] = customer[NAME_COL].str[:1]

    customer["是否雙重命中"] = np.where(
        (customer[SLEEP_FLAG] == 1) & (customer[LUMP_FLAG] == 1), 1, 0
    )

    customer["郵遞區號_排序用"] = pd.to_numeric(customer[ZIP_COL], errors="coerce")
    customer["地區排序輔助"] = customer[DISTRICT_COL].fillna("ZZZ")

    result_list = []

    for agent, g in customer.groupby(AGENT_COL):

        g = g.sort_values(
            ["是否雙重命中", "郵遞區號_排序用", "地區排序輔助", "姓氏", CUST_COL],
            ascending=[False, True, True, True, True]
        ).reset_index(drop=True)

        g["名單排序順位"] = np.arange(1, len(g) + 1)

        # ===== 躉投 =====
        g["躉投排序順位"] = np.where(
            g[LUMP_FLAG] == 1,
            g[LUMP_FLAG].cumsum(),
            np.nan
        )

        g["躉投批次"] = ((g["躉投排序順位"] - 1) // BATCH_SIZE) + 1

        # ===== 靜止 =====
        g["靜止戶排序順位"] = np.where(
            g[SLEEP_FLAG] == 1,
            g[SLEEP_FLAG].cumsum(),
            np.nan
        )

        g["靜止戶批次"] = ((g["靜止戶排序順位"] - 1) // BATCH_SIZE) + 1

        # ===== 解鎖：只能用已開放批次內的有效拜訪來解鎖下一批 =====
        # ===== 批次總數（先算，等等會用）=====
        lump_total_batches = int(g["躉投批次"].max()) if g["躉投批次"].notna().any() else 0
        sleep_total_batches = int(g["靜止戶批次"].max()) if g["靜止戶批次"].notna().any() else 0

        # 躉投：每一批各自計算有效拜訪
        lump_batch_visit = (
            g[g[LUMP_FLAG] == 1]
            .groupby("躉投批次")["是否含#躉投_row"]
            .sum()
            .to_dict()
        )

        current_open_lump = 1

        while current_open_lump < lump_total_batches:
            visited_cnt = lump_batch_visit.get(current_open_lump, 0)

            if visited_cnt >= UNLOCK_THRESHOLD:
                current_open_lump += 1
            else:
                break

        current_open_lump = min(current_open_lump, max(lump_total_batches, 1))


        # 靜止戶：每一批各自計算有效拜訪
        sleep_batch_visit = (
            g[g[SLEEP_FLAG] == 1]
            .groupby("靜止戶批次")["是否含#靜止_row"]
            .sum()
            .to_dict()
        )

        current_open_sleep = 1

        while current_open_sleep < sleep_total_batches:
            visited_cnt = sleep_batch_visit.get(current_open_sleep, 0)

            if visited_cnt >= UNLOCK_THRESHOLD:
                current_open_sleep += 1
            else:
                break

        current_open_sleep = min(current_open_sleep, max(sleep_total_batches, 1))


        g["目前開放躉投批次"] = current_open_lump
        g["目前開放靜止戶批次"] = current_open_sleep

        # ===== 是否釋出 =====
        g["是否釋出"] = np.where(
            ((g[LUMP_FLAG] == 1) & (g["躉投批次"] <= g["目前開放躉投批次"])) |
            ((g[SLEEP_FLAG] == 1) & (g["靜止戶批次"] <= g["目前開放靜止戶批次"])),
            1, 0
        )

        # ===== KPI =====
        current_lump_visit = lump_batch_visit.get(current_open_lump, 0)
        current_sleep_visit = sleep_batch_visit.get(current_open_sleep, 0)

        g["躉投有效拜訪客戶數"] = current_lump_visit
        g["靜止戶有效拜訪客戶數"] = current_sleep_visit
        g["有效拜訪客戶數"] = current_lump_visit + current_sleep_visit

        g["躉投距離解鎖下一批"] = max(UNLOCK_THRESHOLD - current_lump_visit, 0)
        g["靜止戶距離解鎖下一批"] = max(UNLOCK_THRESHOLD - current_sleep_visit, 0)

        g["距離解鎖下一批"] = (
            g["躉投距離解鎖下一批"] + g["靜止戶距離解鎖下一批"]
        )

        # g["躉投批次顯示"] = g["目前開放躉投批次"].astype(str) + "/?"
        # g["靜止戶批次顯示"] = g["目前開放靜止戶批次"].astype(str) + "/?"

        lump_total_batches = int(g["躉投批次"].max()) if g["躉投批次"].notna().any() else 0
        sleep_total_batches = int(g["靜止戶批次"].max()) if g["靜止戶批次"].notna().any() else 0

        # g["躉投批次顯示"] = (
        #     g["目前開放躉投批次"].astype(str) + "/" + str(lump_total_batches)
        # )

        # g["靜止戶批次顯示"] = (
        #     g["目前開放靜止戶批次"].astype(str) + "/" + str(sleep_total_batches)
        # )

        g["躉投批次顯示"] = np.where(
            lump_total_batches == 0,
            "無躉投名單",
            g["目前開放躉投批次"].astype(str) + "/" + str(lump_total_batches)
        )

        g["靜止戶批次顯示"] = np.where(
            sleep_total_batches == 0,
            "無靜止名單",
            g["目前開放靜止戶批次"].astype(str) + "/" + str(sleep_total_batches)
        )

        
        g["總批次"] = max(lump_total_batches, sleep_total_batches)

        result_list.append(g)

    result = pd.concat(result_list)

    result["是否有效拜訪"] = result["是否有效拜訪_row"]
    result["是否含#躉投"] = result["是否含#躉投_row"]
    result["是否含#靜止"] = result["是否含#靜止_row"]
    result["有效拜訪時間"] = result["有效拜訪時間_row"].dt.strftime("%Y-%m-%d %H:%M:%S").fillna("")

    # ===== 避免 NaTType 無法 JSON serializable =====
    if "有效拜訪時間_row" in result.columns:
        result = result.drop(columns=["有效拜訪時間_row"])

    if "拜訪時間_解析" in result.columns:
        result = result.drop(columns=["拜訪時間_解析"])

    result["有效拜訪時間"] = pd.to_datetime(
        result["有效拜訪時間"],
        errors="coerce"
    ).dt.strftime("%Y-%m-%d %H:%M:%S").fillna("")

    return result