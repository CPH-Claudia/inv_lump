# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 09:03:51 2026

@author: Z01788
"""
# %% packages
import pantab
import pandas as pd
import numpy as np

# %% raw df
NEEDED_COLS = [
    # 核心ID
    "保單申請案號", "被保人身分證字號", "要保人身分證字號",

    # 日期
    "投保日", "被保人生日",

    # 客戶屬性
    "被保人性別", 
    "(要)保人-家庭年收入(萬)", "(要)保人-工作年收入(萬)", "(要)保人-其他年收入(萬)", 

    # 商品 / 保單結構
    "商品名稱", "商品系統代碼", "商品險種主類別", "商品險種次類別",
    "型別/計劃別", "主附約別", "繳別", "保單狀況", "產壽險別",
    "繳費期間(起)", "繳費期間(迄)", 

    # 金額
    "繳款保費", "繳款FYC",

    # 組織 / 歸屬
    "經紀人1", "經紀人1業代", "營業單位", "保險公司", "保險公司代碼"
]

hyper_path = r"D:\投資型\lump\ipo_0331.hyper"
df_raw = pantab.frame_from_hyper(
    hyper_path,
    table=("Extract", "Extract")
)
existing_cols = [c for c in NEEDED_COLS if c in df_raw.columns]
missing_cols = [c for c in NEEDED_COLS if c not in df_raw.columns]
print("讀入欄位數:", len(existing_cols))
print("缺少欄位:", missing_cols)
df = df_raw[existing_cols].copy()

# 計算「要保人年收入」
# 確認三個欄位原本就在 NEEDED_COLS 裡
df["要保人年收入"] = (
    df["(要)保人-家庭年收入(萬)"]
    .where(df["(要)保人-家庭年收入(萬)"] > 0)
    .fillna(df["(要)保人-工作年收入(萬)"].where(df["(要)保人-工作年收入(萬)"] > 0))
    .fillna(df["(要)保人-其他年收入(萬)"].where(df["(要)保人-其他年收入(萬)"] > 0))
)

cols_to_remove = ["(要)保人-家庭年收入(萬)", "(要)保人-工作年收入(萬)", "(要)保人-其他年收入(萬)"]
df.drop(columns=cols_to_remove, errors='ignore', inplace=True)

print("資料筆數:", len(df))


# %% policy df
def build_policy_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    將保單明細層資料整理成 policy-level table（一張保單一列）

    回傳：
    - policy_df：每張保單一列

    這版額外新增：
    - policy-level 產壽險拆分件數 / 保費 / FYC 欄位
    - 可供後續 customer / benchmark / candidate 直接 groupby sum 使用
    """

    df = df.copy()

    # =========================
    # 1. 基本清理
    # =========================
    df.columns = df.columns.str.strip()

    str_cols = [
        "保單申請案號", "被保人身分證字號", "要保人身分證字號",
        "商品名稱", "商品系統代碼", "商品險種主類別", "商品險種次類別",
        "型別/計劃別", "主附約別", "繳別", "保單狀況",
        "經紀人1", "經紀人1業代", "營業單位", "產壽險別",
        "保險公司", "保險公司代碼", "被保人性別"
    ]

    for c in str_cols:
        if c in df.columns:
            df[c] = df[c].astype("string").str.strip()
            df[c] = df[c].replace({
                "": pd.NA,
                "nan": pd.NA,
                "None": pd.NA,
                "NaT": pd.NA
            })

    # 日期欄位
    date_cols = ["投保日", "被保人生日", "繳費期間(起)", "繳費期間(迄)"]
    for c in date_cols:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")

    # 數值欄位
    num_cols = ["繳款保費", "繳款FYC", "要保人年收入", "繳費年期"]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # 排除沒有保單申請案號的資料
    df = df[df["保單申請案號"].notna()].copy()

    # =========================
    # 1.1 統一產壽險別標準
    # =========================
    if "產壽險別" in df.columns:
        df["產壽險別"] = df["產壽險別"].replace({
            "壽": "壽險",
            "人壽": "壽險",
            "壽保": "壽險",
            "產": "產險",
            "財產": "產險",
            "財險": "產險"
        })

    # =========================
    # 1.5. 重算投保年齡
    # =========================
    if {"投保日", "被保人生日"}.issubset(df.columns):
        df["被保人投保年齡_重算"] = np.floor(
            (df["投保日"] - df["被保人生日"]).dt.days / 365.25
        )

    # =========================
    # 2. 明細層 flags
    # =========================
    df["是否主約"] = (df["主附約別"] == "主約").astype("Int64")
    df["是否附約"] = (df["主附約別"] == "附約").astype("Int64")

    df["是否投資型明細"] = (df["商品險種主類別"] == "投資型").astype("Int64")
    df["是否躉繳明細"] = (df["繳別"] == "躉繳").astype("Int64")
    df["是否躉繳投資型"] = (
        (df["商品險種主類別"] == "投資型") &
        (df["繳別"] == "躉繳")
    ).astype("Int64")

    df["要被保人是否同一人"] = (
        df["被保人身分證字號"].fillna("") == df["要保人身分證字號"].fillna("")
    ).astype("Int64")

    # 明細層產壽險拆分欄位（後續可回頭查）
    df["明細保費_壽險"] = np.where(df["產壽險別"] == "壽險", df["繳款保費"], 0)
    df["明細保費_產險"] = np.where(df["產壽險別"] == "產險", df["繳款保費"], 0)
    df["明細繳款FYC_壽險"] = np.where(df["產壽險別"] == "壽險", df["繳款FYC"], 0)
    df["明細繳款FYC_產險"] = np.where(df["產壽險別"] == "產險", df["繳款FYC"], 0)

    # =========================
    # 3. 找主約資料
    #    若同保單有多筆主約，保留排序後第一筆
    # =========================
    sort_cols = ["保單申請案號"]
    ascending_list = [True]

    if "投保日" in df.columns:
        sort_cols.append("投保日")
        ascending_list.append(True)

    if "繳款保費" in df.columns:
        sort_cols.append("繳款保費")
        ascending_list.append(False)

    df_sorted = df.sort_values(sort_cols, ascending=ascending_list).copy()
    main_df = df_sorted[df_sorted["是否主約"] == 1].copy()

    main_cols = [
        "保單申請案號",
        "商品名稱",
        "商品系統代碼",
        "商品險種主類別",
        "商品險種次類別",
        "型別/計劃別",
        "繳別",
        "繳款保費",
        "繳款FYC"
    ]
    main_cols = [c for c in main_cols if c in main_df.columns]

    main_df = (
        main_df[main_cols]
        .drop_duplicates(subset=["保單申請案號"], keep="first")
        .rename(columns={
            "商品名稱": "主約商品名稱",
            "商品系統代碼": "主約商品系統代碼",
            "商品險種主類別": "主約商品險種主類別",
            "商品險種次類別": "主約商品險種次類別",
            "型別/計劃別": "主約型別/計劃別",
            "繳別": "主約繳別",
            "繳款保費": "主約保費",
            "繳款FYC": "主約繳款FYC"
        })
    )

    # =========================
    # 4. 保單層聚合
    # =========================
    agg_dict = {
        "被保人身分證字號": "first",
        "要保人身分證字號": "first",
        "投保日": "min",
        "保單狀況": "first",
        "經紀人1業代": "first",
        "營業單位": "first",

        "是否主約": "sum",
        "是否附約": "sum",
        "繳款保費": "sum",
        "繳款FYC": "sum",

        "商品名稱": pd.Series.nunique,
        "商品險種主類別": pd.Series.nunique,
        "商品險種次類別": pd.Series.nunique,

        "是否投資型明細": "max",
        "是否躉繳明細": "max",
        "是否躉繳投資型": "max",
        "要被保人是否同一人": "max",

        # 明細層產壽險拆分加總
        "明細保費_壽險": "sum",
        "明細保費_產險": "sum",
        "明細繳款FYC_壽險": "sum",
        "明細繳款FYC_產險": "sum",
    }

    optional_first_cols = [
        "產壽險別", "保險公司", "保險公司代碼",
        "繳費年期", "繳費期間(起)", "繳費期間(迄)",
        "被保人性別", "被保人投保年齡_重算",
        "要保人年收入", "被保人生日"
    ]

    for c in optional_first_cols:
        if c in df.columns:
            agg_dict[c] = "first"

    policy_df = (
        df.groupby("保單申請案號", dropna=False)
        .agg(agg_dict)
        .reset_index()
        .rename(columns={
            "繳款保費": "保單總保費",
            "繳款FYC": "保單總繳款FYC",
            "商品名稱": "商品名稱數",
            "商品險種主類別": "商品險種主類別數",
            "商品險種次類別": "商品險種次類別數",
            "是否主約": "主約筆數",
            "是否附約": "附約筆數",
            "是否躉繳投資型": "是否含躉繳投資型",
        })
    )

    # 再另外補「明細列數」
    detail_cnt = (
        df.groupby("保單申請案號", dropna=False)
        .size()
        .reset_index(name="明細列數")
    )
    policy_df = policy_df.merge(detail_cnt, on="保單申請案號", how="left")

    # =========================
    # 4.5 保單層產壽險件數欄位
    # =========================
    # 一張保單一列，因此這裡做成「後續可加總」的欄位
    policy_df["保單件數_壽險"] = (policy_df["產壽險別"] == "壽險").astype("Int64")
    policy_df["保單件數_產險"] = (policy_df["產壽險別"] == "產險").astype("Int64")

    policy_df["保單總保費_壽險"] = np.where(
        policy_df["產壽險別"] == "壽險",
        policy_df["保單總保費"],
        0
    )
    policy_df["保單總保費_產險"] = np.where(
        policy_df["產壽險別"] == "產險",
        policy_df["保單總保費"],
        0
    )

    policy_df["保單總繳款FYC_壽險"] = np.where(
        policy_df["產壽險別"] == "壽險",
        policy_df["保單總繳款FYC"],
        0
    )
    policy_df["保單總繳款FYC_產險"] = np.where(
        policy_df["產壽險別"] == "產險",
        policy_df["保單總繳款FYC"],
        0
    )

    # =========================
    # 5. merge 主約資訊
    # =========================
    policy_df = policy_df.merge(main_df, on="保單申請案號", how="left")

    # =========================
    # 6. 建立保單層判斷欄位
    # =========================
    policy_df["主約是否投資型"] = (
        policy_df["主約商品險種主類別"] == "投資型"
    ).astype("Int64")

    policy_df["主約是否躉繳"] = (
        policy_df["主約繳別"] == "躉繳"
    ).astype("Int64")

    policy_df["主約是否躉繳投資型"] = (
        (policy_df["主約商品險種主類別"] == "投資型") &
        (policy_df["主約繳別"] == "躉繳")
    ).astype("Int64")

    # 無主約時，保留是否含躉繳投資型作為輔助
    policy_df["保單是否躉繳投資型"] = policy_df["主約是否躉繳投資型"].copy()
    mask_no_main = policy_df["主約商品名稱"].isna()
    policy_df.loc[mask_no_main, "保單是否躉繳投資型"] = policy_df.loc[mask_no_main, "是否含躉繳投資型"]

    # =========================
    # 7. 一些常用衍生欄位
    # =========================
    policy_df["是否只有主約"] = (
        (policy_df["主約筆數"] >= 1) & (policy_df["附約筆數"] == 0)
    ).astype("Int64")

    policy_df["是否有附約"] = (policy_df["附約筆數"] > 0).astype("Int64")

    numeric_cols = [
        "保單總保費", "保單總繳款FYC", "明細列數",
        "明細保費_壽險", "明細保費_產險",
        "明細繳款FYC_壽險", "明細繳款FYC_產險",
        "保單總保費_壽險", "保單總保費_產險",
        "保單總繳款FYC_壽險", "保單總繳款FYC_產險",
    ]
    for c in numeric_cols:
        if c in policy_df.columns:
            policy_df[c] = pd.to_numeric(policy_df[c], errors="coerce")

    policy_df["平均每明細保費"] = policy_df["保單總保費"] / policy_df["明細列數"]
    policy_df["平均每明細繳款FYC"] = policy_df["保單總繳款FYC"] / policy_df["明細列數"]

    # 依產壽險拆分的平均值（可選）
    policy_df["平均每明細保費_壽險"] = policy_df["明細保費_壽險"] / policy_df["明細列數"]
    policy_df["平均每明細保費_產險"] = policy_df["明細保費_產險"] / policy_df["明細列數"]
    policy_df["平均每明細繳款FYC_壽險"] = policy_df["明細繳款FYC_壽險"] / policy_df["明細列數"]
    policy_df["平均每明細繳款FYC_產險"] = policy_df["明細繳款FYC_產險"] / policy_df["明細列數"]

    # =========================
    # 8. 排序與保單序號
    # =========================
    sort_cols = ["被保人身分證字號", "投保日", "保單申請案號"]
    sort_cols = [c for c in sort_cols if c in policy_df.columns]
    policy_df = policy_df.sort_values(sort_cols).reset_index(drop=True)

    if "被保人身分證字號" in policy_df.columns:
        policy_df["保單序號"] = policy_df.groupby("被保人身分證字號").cumcount() + 1

    # =========================
    # 9. 欄位順序整理
    # =========================
    front_cols = [
        "保單申請案號",
        "被保人身分證字號",
        "要保人身分證字號",
        "投保日",
        "保單序號" if "保單序號" in policy_df.columns else None,
        "營業單位",
        "經紀人1業代",
        "保單狀況",
        "產壽險別",

        "主約商品名稱",
        "主約商品系統代碼" if "主約商品系統代碼" in policy_df.columns else None,
        "主約商品險種主類別",
        "主約商品險種次類別",
        "主約型別/計劃別" if "主約型別/計劃別" in policy_df.columns else None,
        "主約繳別",
        "主約保費",
        "主約繳款FYC",

        "明細列數",
        "主約筆數",
        "附約筆數",
        "商品名稱數",
        "商品險種主類別數",
        "商品險種次類別數",

        "保單件數_壽險",
        "保單件數_產險",
        "保單總保費",
        "保單總保費_壽險",
        "保單總保費_產險",
        "保單總繳款FYC",
        "保單總繳款FYC_壽險",
        "保單總繳款FYC_產險",
        "明細保費_壽險",
        "明細保費_產險",
        "明細繳款FYC_壽險",
        "明細繳款FYC_產險",

        "是否投資型明細",
        "是否躉繳明細",
        "是否含躉繳投資型",
        "主約是否投資型",
        "主約是否躉繳",
        "主約是否躉繳投資型",
        "保單是否躉繳投資型",
        "是否只有主約",
        "是否有附約",
        "要被保人是否同一人"
    ]
    front_cols = [c for c in front_cols if c is not None and c in policy_df.columns]

    other_cols = [c for c in policy_df.columns if c not in front_cols]
    policy_df = policy_df[front_cols + other_cols]

    return policy_df

policy_df = build_policy_table(df)

# print(policy_df.shape)
# print(policy_df.head())
# print(policy_df.columns.tolist())

# %% customer df
def build_customer_table(policy_df: pd.DataFrame, analysis_date=None) -> pd.DataFrame:
    """
    將 policy-level table（一張保單一列）整理成 customer-level table（一位客戶一列）

    參數
    ----------
    policy_df : pd.DataFrame
        每張保單一列的資料表
    analysis_date : str / pd.Timestamp / None
        分析基準日；若為 None，預設使用 policy_df['投保日'] 的最大值

    回傳
    ----------
    customer_df : pd.DataFrame
        每位客戶一列的客戶層級特徵表
    """

    df = policy_df.copy()

    # =========================
    # 0. 檢查必要欄位
    # =========================
    required_cols = [
        "被保人身分證字號",
        "保單申請案號",
        "投保日",
        "保單總保費",
        "保單總繳款FYC"
    ]
    missing_required = [c for c in required_cols if c not in df.columns]
    if missing_required:
        raise ValueError(f"policy_df 缺少必要欄位: {missing_required}")

    # =========================
    # 1. 基本清理 / 型別整理
    # =========================
    df.columns = df.columns.str.strip()

    if "投保日" in df.columns:
        df["投保日"] = pd.to_datetime(df["投保日"], errors="coerce")
    if "被保人生日" in df.columns:
        df["被保人生日"] = pd.to_datetime(df["被保人生日"], errors="coerce")

    # # 若 policy_df 尚未有產壽險拆分欄位，補一版保底
    # if "保單件數_壽險" not in df.columns:
    #     df["保單件數_壽險"] = np.where(df.get("產壽險別", pd.Series(index=df.index)) == "壽險", 1, 0)
    # if "保單件數_產險" not in df.columns:
    #     df["保單件數_產險"] = np.where(df.get("產壽險別", pd.Series(index=df.index)) == "產險", 1, 0)

    # if "保單總保費_壽險" not in df.columns:
    #     df["保單總保費_壽險"] = np.where(df.get("產壽險別", pd.Series(index=df.index)) == "壽險", df["保單總保費"], 0)
    # if "保單總保費_產險" not in df.columns:
    #     df["保單總保費_產險"] = np.where(df.get("產壽險別", pd.Series(index=df.index)) == "產險", df["保單總保費"], 0)

    # if "保單總繳款FYC_壽險" not in df.columns:
    #     df["保單總繳款FYC_壽險"] = np.where(df.get("產壽險別", pd.Series(index=df.index)) == "壽險", df["保單總繳款FYC"], 0)
    # if "保單總繳款FYC_產險" not in df.columns:
    #     df["保單總繳款FYC_產險"] = np.where(df.get("產壽險別", pd.Series(index=df.index)) == "產險", df["保單總繳款FYC"], 0)

    num_cols = [
        "保單序號",
        "主約保費",
        "保單總保費",
        "保單總繳款FYC",
        "保單件數_壽險",
        "保單件數_產險",
        "保單總保費_壽險",
        "保單總保費_產險",
        "保單總繳款FYC_壽險",
        "保單總繳款FYC_產險",
        "是否含躉繳投資型",
        "主約是否躉繳投資型",
        "保單是否躉繳投資型",
        "要被保人是否同一人",
        "是否有附約" if "是否有附約" in df.columns else None,
        "被保人投保年齡_重算",
        "要保人年收入",
    ]
    num_cols = [c for c in num_cols if c is not None and c in df.columns]

    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    sort_cols = [c for c in ["被保人身分證字號", "投保日", "保單申請案號"] if c in df.columns]
    df = df.sort_values(sort_cols).reset_index(drop=True)

    # =========================
    # 2. analysis_date
    # =========================
    if analysis_date is None:
        analysis_date = df["投保日"].max()
    else:
        analysis_date = pd.to_datetime(analysis_date)

    # =========================
    # 3. 若保單序號不存在，補一個
    # =========================
    if "保單序號" not in df.columns:
        df["保單序號"] = df.groupby("被保人身分證字號").cumcount() + 1

    # =========================
    # 3.5. 重算目前年齡
    # =========================
    if "被保人生日" in df.columns:
        df["被保人目前年齡_重算"] = np.floor(
            (analysis_date - df["被保人生日"]).dt.days / 365.25
        )

    # =========================
    # 4. 時間旗標
    # =========================
    df["近1年保單"] = (df["投保日"] >= (analysis_date - pd.Timedelta(days=365))).astype("Int64")
    df["近2年保單"] = (df["投保日"] >= (analysis_date - pd.Timedelta(days=730))).astype("Int64")
    df["近3年保單"] = (df["投保日"] >= (analysis_date - pd.Timedelta(days=1095))).astype("Int64")

    # 近年 × 產壽險拆分
    df["近1年保單_壽險"] = df["近1年保單"] * df["保單件數_壽險"]
    df["近1年保單_產險"] = df["近1年保單"] * df["保單件數_產險"]
    df["近2年保單_壽險"] = df["近2年保單"] * df["保單件數_壽險"]
    df["近2年保單_產險"] = df["近2年保單"] * df["保單件數_產險"]
    df["近3年保單_壽險"] = df["近3年保單"] * df["保單件數_壽險"]
    df["近3年保單_產險"] = df["近3年保單"] * df["保單件數_產險"]

    # =========================
    # 5. 高保費旗標（以全體 P75 / P90）
    # =========================
    premium_base = df["保單總保費"].dropna()
    p75 = premium_base.quantile(0.75) if len(premium_base) > 0 else np.nan
    p90 = premium_base.quantile(0.90) if len(premium_base) > 0 else np.nan

    df["是否高保費保單_P75"] = (df["保單總保費"] >= p75).astype("Int64") if pd.notna(p75) else 0
    df["是否高保費保單_P90"] = (df["保單總保費"] >= p90).astype("Int64") if pd.notna(p90) else 0

    # 高保費 × 產壽險拆分
    df["高保費保單_P75_壽險"] = df["是否高保費保單_P75"] * df["保單件數_壽險"]
    df["高保費保單_P75_產險"] = df["是否高保費保單_P75"] * df["保單件數_產險"]
    df["高保費保單_P90_壽險"] = df["是否高保費保單_P90"] * df["保單件數_壽險"]
    df["高保費保單_P90_產險"] = df["是否高保費保單_P90"] * df["保單件數_產險"]

    # =========================
    # 6. 各主類別旗標（若有值）
    # =========================
    if "主約商品險種主類別" in df.columns:
        df["主約_壽險"] = (df["主約商品險種主類別"] == "壽險").astype("Int64")
        df["主約_健康險"] = (df["主約商品險種主類別"] == "健康險").astype("Int64")
        df["主約_傷害險"] = (df["主約商品險種主類別"] == "傷害險").astype("Int64")
        df["主約_投資型"] = (df["主約商品險種主類別"] == "投資型").astype("Int64")
        df["主約_年金險"] = (df["主約商品險種主類別"] == "年金險").astype("Int64")
    else:
        for c in ["主約_壽險", "主約_健康險", "主約_傷害險", "主約_投資型", "主約_年金險"]:
            df[c] = 0

    # =========================
    # 7. 前一張保單間隔
    # =========================
    df["前一張投保日"] = df.groupby("被保人身分證字號")["投保日"].shift(1)
    df["保單間隔天數"] = (df["投保日"] - df["前一張投保日"]).dt.days

    # =========================
    # 8. 客戶基本資料彙整函式
    # =========================
    def mode_or_nan(s):
        s = s.dropna()
        if s.empty:
            return np.nan
        m = s.mode()
        if len(m) == 0:
            return np.nan
        return m.iloc[0]

    def last_valid(s):
        s = s.dropna()
        if s.empty:
            return np.nan
        return s.iloc[-1]

    # =========================
    # 9. 客戶層主要聚合
    # =========================
    agg_dict = {
        "保單申請案號": "count",
        "投保日": ["min", "max"],

        "保單件數_壽險": "sum",
        "保單件數_產險": "sum",

        "保單總保費": ["sum", "mean", "median", "max"],
        "保單總保費_壽險": ["sum", "mean", "median", "max"],
        "保單總保費_產險": ["sum", "mean", "median", "max"],

        "保單總繳款FYC": ["sum", "mean", "max"],
        "保單總繳款FYC_壽險": ["sum", "mean", "max"],
        "保單總繳款FYC_產險": ["sum", "mean", "max"],

        "主約保費": ["mean", "median", "max"] if "主約保費" in df.columns else ["count"],

        "是否含躉繳投資型": "sum" if "是否含躉繳投資型" in df.columns else "count",
        "主約是否躉繳投資型": "sum" if "主約是否躉繳投資型" in df.columns else "count",
        "保單是否躉繳投資型": "sum" if "保單是否躉繳投資型" in df.columns else "count",

        "近1年保單": "sum",
        "近2年保單": "sum",
        "近3年保單": "sum",
        "近1年保單_壽險": "sum",
        "近1年保單_產險": "sum",
        "近2年保單_壽險": "sum",
        "近2年保單_產險": "sum",
        "近3年保單_壽險": "sum",
        "近3年保單_產險": "sum",

        "是否高保費保單_P75": "sum",
        "是否高保費保單_P90": "sum",
        "高保費保單_P75_壽險": "sum",
        "高保費保單_P75_產險": "sum",
        "高保費保單_P90_壽險": "sum",
        "高保費保單_P90_產險": "sum",

        "主約_壽險": "sum",
        "主約_健康險": "sum",
        "主約_傷害險": "sum",
        "主約_投資型": "sum",
        "主約_年金險": "sum",

        "保單間隔天數": ["mean", "median", "min", "max"],
        "要被保人是否同一人": "mean" if "要被保人是否同一人" in df.columns else "count",
    }

    if "主約商品險種主類別" in df.columns:
        agg_dict["主約商品險種主類別"] = pd.Series.nunique
    if "主約商品險種次類別" in df.columns:
        agg_dict["主約商品險種次類別"] = pd.Series.nunique
    if "主約商品名稱" in df.columns:
        agg_dict["主約商品名稱"] = pd.Series.nunique
    if "主約繳別" in df.columns:
        agg_dict["主約繳別"] = pd.Series.nunique
    if "營業單位" in df.columns:
        agg_dict["營業單位"] = last_valid
    if "經紀人1業代" in df.columns:
        agg_dict["經紀人1業代"] = last_valid
    if "保單狀況" in df.columns:
        agg_dict["保單狀況"] = last_valid
    if "被保人性別" in df.columns:
        agg_dict["被保人性別"] = mode_or_nan
    if "被保人投保年齡_重算" in df.columns:
        agg_dict["被保人投保年齡_重算"] = ["min", "max", "mean"]
    if "被保人目前年齡_重算" in df.columns:
        agg_dict["被保人目前年齡_重算"] = last_valid
    if "要保人年收入" in df.columns:
        agg_dict["要保人年收入"] = "max"
    if "被保人生日" in df.columns:
        agg_dict["被保人生日"] = "first"

    customer_df = df.groupby("被保人身分證字號", dropna=False).agg(agg_dict)

    # =========================
    # 10. 展平 MultiIndex 欄位
    # =========================
    def flatten_columns(columns):
        new_cols = []
        for col in columns:
            if isinstance(col, tuple):
                parts = [str(x) for x in col if x not in ("", None)]
                new_cols.append("_".join(parts))
            else:
                new_cols.append(str(col))
        return new_cols

    customer_df.columns = flatten_columns(customer_df.columns)
    customer_df = customer_df.reset_index()

    # =========================
    # 11. 欄位重新命名
    # =========================
    rename_map = {
        "保單申請案號_count": "保單數",
        "投保日_min": "首次投保日",
        "投保日_max": "最近投保日",

        "保單件數_壽險_sum": "壽險保單數",
        "保單件數_產險_sum": "產險保單數",

        "保單總保費_sum": "累計保單總保費",
        "保單總保費_mean": "平均每張保單保費",
        "保單總保費_median": "保單保費中位數",
        "保單總保費_max": "最大單張保單保費",

        "保單總保費_壽險_sum": "累計壽險保單總保費",
        "保單總保費_壽險_mean": "平均每張壽險保單保費",
        "保單總保費_壽險_median": "壽險保單保費中位數",
        "保單總保費_壽險_max": "最大單張壽險保單保費",

        "保單總保費_產險_sum": "累計產險保單總保費",
        "保單總保費_產險_mean": "平均每張產險保單保費",
        "保單總保費_產險_median": "產險保單保費中位數",
        "保單總保費_產險_max": "最大單張產險保單保費",

        "保單總繳款FYC_sum": "累計保單總繳款FYC",
        "保單總繳款FYC_mean": "平均每張保單繳款FYC",
        "保單總繳款FYC_max": "最大單張保單繳款FYC",

        "保單總繳款FYC_壽險_sum": "累計壽險保單總繳款FYC",
        "保單總繳款FYC_壽險_mean": "平均每張壽險保單繳款FYC",
        "保單總繳款FYC_壽險_max": "最大單張壽險保單繳款FYC",

        "保單總繳款FYC_產險_sum": "累計產險保單總繳款FYC",
        "保單總繳款FYC_產險_mean": "平均每張產險保單繳款FYC",
        "保單總繳款FYC_產險_max": "最大單張產險保單繳款FYC",

        "主約保費_mean": "平均主約保費",
        "主約保費_median": "主約保費中位數",
        "主約保費_max": "最大主約保費",

        "是否含躉繳投資型_sum": "含躉繳投資型保單數",
        "主約是否躉繳投資型_sum": "主約躉繳投資型保單數",
        "保單是否躉繳投資型_sum": "躉繳投資型保單數",

        "近1年保單_sum": "近1年保單數",
        "近2年保單_sum": "近2年保單數",
        "近3年保單_sum": "近3年保單數",
        "近1年保單_壽險_sum": "近1年壽險保單數",
        "近1年保單_產險_sum": "近1年產險保單數",
        "近2年保單_壽險_sum": "近2年壽險保單數",
        "近2年保單_產險_sum": "近2年產險保單數",
        "近3年保單_壽險_sum": "近3年壽險保單數",
        "近3年保單_產險_sum": "近3年產險保單數",

        "是否高保費保單_P75_sum": "高保費保單數_P75",
        "是否高保費保單_P90_sum": "高保費保單數_P90",
        "高保費保單_P75_壽險_sum": "高保費壽險保單數_P75",
        "高保費保單_P75_產險_sum": "高保費產險保單數_P75",
        "高保費保單_P90_壽險_sum": "高保費壽險保單數_P90",
        "高保費保單_P90_產險_sum": "高保費產險保單數_P90",

        "主約_壽險_sum": "主約壽險保單數",
        "主約_健康險_sum": "主約健康險保單數",
        "主約_傷害險_sum": "主約傷害險保單數",
        "主約_投資型_sum": "主約投資型保單數_依主類別",
        "主約_年金險_sum": "主約年金險保單數",

        "保單間隔天數_mean": "平均保單間隔天數",
        "保單間隔天數_median": "保單間隔天數中位數",
        "保單間隔天數_min": "最短保單間隔天數",
        "保單間隔天數_max": "最長保單間隔天數",

        "要被保人是否同一人_mean": "要被保人同一人比例",

        "主約商品險種主類別_nunique": "主約商品險種主類別數",
        "主約商品險種次類別_nunique": "主約商品險種次類別數",
        "主約商品名稱_nunique": "主約商品名稱數",
        "主約繳別_nunique": "主約繳別類型數",

        "營業單位_last_valid": "目前營業單位",
        "經紀人1業代_last_valid": "目前經紀人1業代",
        "保單狀況_last_valid": "最近保單狀況",
        "被保人性別_mode_or_nan": "被保人性別",

        "被保人投保年齡_重算_min": "最小投保年齡_重算",
        "被保人投保年齡_重算_max": "最大投保年齡_重算",
        "被保人投保年齡_重算_mean": "平均投保年齡_重算",
        "被保人目前年齡_重算_last_valid": "被保人目前年齡_重算",

        "要保人年收入_max": "要保人年收入",
        "被保人生日_first": "被保人生日",
    }

    customer_df = customer_df.rename(columns=rename_map)

    # =========================
    # 12. 衍生欄位
    # =========================
    customer_df["投保年資天數"] = (analysis_date - customer_df["首次投保日"]).dt.days
    customer_df["距離最近投保天數"] = (analysis_date - customer_df["最近投保日"]).dt.days

    # 占比欄位
    # 以總保單數為分母
    ratio_pairs_total = [
        ("壽險保單數", "壽險保單占比"),
        ("產險保單數", "產險保單占比"),
        ("含躉繳投資型保單數", "含躉繳投資型保單比例"),
        ("主約躉繳投資型保單數", "主約躉繳投資型保單比例"),
        ("躉繳投資型保單數", "躉繳投資型保單比例"),
        ("近1年保單數", "近1年保單比例"),
        ("近2年保單數", "近2年保單比例"),
        ("近3年保單數", "近3年保單比例"),
        ("高保費保單數_P75", "高保費保單比例_P75"),
        ("高保費保單數_P90", "高保費保單比例_P90"),
        ("主約壽險保單數", "主約壽險保單比例"),
        ("主約健康險保單數", "主約健康險保單比例"),
        ("主約傷害險保單數", "主約傷害險保單比例"),
        ("主約投資型保單數_依主類別", "主約投資型保單比例_依主類別"),
        ("主約年金險保單數", "主約年金險保單比例"),
        ("近1年壽險保單數", "近1年壽險保單比例"),
        ("近1年產險保單數", "近1年產險保單比例"),
        ("近2年壽險保單數", "近2年壽險保單比例"),
        ("近2年產險保單數", "近2年產險保單比例"),
        ("近3年壽險保單數", "近3年壽險保單比例"),
        ("近3年產險保單數", "近3年產險保單比例"),
        ("高保費壽險保單數_P75", "高保費壽險保單比例_P75"),
        ("高保費產險保單數_P75", "高保費產險保單比例_P75"),
        ("高保費壽險保單數_P90", "高保費壽險保單比例_P90"),
        ("高保費產險保單數_P90", "高保費產險保單比例_P90"),
    ]

    for num_col, ratio_col in ratio_pairs_total:
        if num_col in customer_df.columns:
            customer_df[ratio_col] = np.where(
                customer_df["保單數"] > 0,
                customer_df[num_col] / customer_df["保單數"],
                np.nan
            )

    # 保費 / FYC 結構占比
    if {"累計保單總保費", "累計壽險保單總保費"}.issubset(customer_df.columns):
        customer_df["壽險保費占比"] = np.where(
            customer_df["累計保單總保費"] > 0,
            customer_df["累計壽險保單總保費"] / customer_df["累計保單總保費"],
            np.nan
        )
    if {"累計保單總保費", "累計產險保單總保費"}.issubset(customer_df.columns):
        customer_df["產險保費占比"] = np.where(
            customer_df["累計保單總保費"] > 0,
            customer_df["累計產險保單總保費"] / customer_df["累計保單總保費"],
            np.nan
        )

    if {"累計保單總繳款FYC", "累計壽險保單總繳款FYC"}.issubset(customer_df.columns):
        customer_df["壽險繳款FYC占比"] = np.where(
            customer_df["累計保單總繳款FYC"] > 0,
            customer_df["累計壽險保單總繳款FYC"] / customer_df["累計保單總繳款FYC"],
            np.nan
        )
    if {"累計保單總繳款FYC", "累計產險保單總繳款FYC"}.issubset(customer_df.columns):
        customer_df["產險繳款FYC占比"] = np.where(
            customer_df["累計保單總繳款FYC"] > 0,
            customer_df["累計產險保單總繳款FYC"] / customer_df["累計保單總繳款FYC"],
            np.nan
        )

    # 是否曾經...
    ever_pairs = [
        ("含躉繳投資型保單數", "是否曾買過含躉繳投資型保單"),
        ("主約躉繳投資型保單數", "是否曾買過主約躉繳投資型"),
        ("躉繳投資型保單數", "是否曾買過躉繳投資型"),
    ]

    for num_col, flag_col in ever_pairs:
        if num_col in customer_df.columns:
            customer_df[flag_col] = (customer_df[num_col] > 0).astype("Int64")

    # 第一張是否就是躉繳投資型
    if "保單是否躉繳投資型" in df.columns:
        first_inv_single = (
            df[df["保單是否躉繳投資型"] == 1]
            .sort_values(["被保人身分證字號", "投保日", "保單申請案號"])
            .groupby("被保人身分證字號", as_index=False)
            .first()[["被保人身分證字號", "保單序號", "投保日"]]
            .rename(columns={
                "保單序號": "首次躉繳投資型保單序號",
                "投保日": "首次躉繳投資型投保日"
            })
        )

        customer_df = customer_df.merge(first_inv_single, on="被保人身分證字號", how="left")

        customer_df["第一張就買躉繳投資型"] = (
            customer_df["首次躉繳投資型保單序號"] == 1
        ).astype("Int64")

    # 收入分級
    if "要保人年收入" in customer_df.columns:
        customer_df["要保人年收入級距"] = pd.cut(
            customer_df["要保人年收入"],
            bins=[0, 50, 100, 200, 500, 1000, np.inf],
            labels=["0-50", "50-100", "100-200", "200-500", "500-1000", "1000+"]
        )

    # =========================
    # 13. 欄位順序整理
    # =========================
    front_cols = [
        "被保人身分證字號",
        "被保人性別" if "被保人性別" in customer_df.columns else None,
        "被保人生日" if "被保人生日" in customer_df.columns else None,
        "被保人目前年齡_重算" if "被保人目前年齡_重算" in customer_df.columns else None,
        "要保人年收入" if "要保人年收入" in customer_df.columns else None,
        "要保人年收入級距" if "要保人年收入級距" in customer_df.columns else None,

        "目前營業單位" if "目前營業單位" in customer_df.columns else None,
        "目前經紀人1業代" if "目前經紀人1業代" in customer_df.columns else None,

        "保單數",
        "壽險保單數" if "壽險保單數" in customer_df.columns else None,
        "產險保單數" if "產險保單數" in customer_df.columns else None,
        "壽險保單占比" if "壽險保單占比" in customer_df.columns else None,
        "產險保單占比" if "產險保單占比" in customer_df.columns else None,

        "首次投保日",
        "最近投保日",
        "投保年資天數",
        "距離最近投保天數",

        "累計保單總保費",
        "累計壽險保單總保費" if "累計壽險保單總保費" in customer_df.columns else None,
        "累計產險保單總保費" if "累計產險保單總保費" in customer_df.columns else None,
        "壽險保費占比" if "壽險保費占比" in customer_df.columns else None,
        "產險保費占比" if "產險保費占比" in customer_df.columns else None,
        "平均每張保單保費",
        "保單保費中位數",
        "最大單張保單保費",
        "平均主約保費" if "平均主約保費" in customer_df.columns else None,
        "主約保費中位數" if "主約保費中位數" in customer_df.columns else None,
        "最大主約保費" if "最大主約保費" in customer_df.columns else None,

        "累計保單總繳款FYC",
        "累計壽險保單總繳款FYC" if "累計壽險保單總繳款FYC" in customer_df.columns else None,
        "累計產險保單總繳款FYC" if "累計產險保單總繳款FYC" in customer_df.columns else None,
        "壽險繳款FYC占比" if "壽險繳款FYC占比" in customer_df.columns else None,
        "產險繳款FYC占比" if "產險繳款FYC占比" in customer_df.columns else None,
        "平均每張保單繳款FYC",
        "最大單張保單繳款FYC",

        "含躉繳投資型保單數" if "含躉繳投資型保單數" in customer_df.columns else None,
        "主約躉繳投資型保單數" if "主約躉繳投資型保單數" in customer_df.columns else None,
        "躉繳投資型保單數" if "躉繳投資型保單數" in customer_df.columns else None,

        "是否曾買過含躉繳投資型保單" if "是否曾買過含躉繳投資型保單" in customer_df.columns else None,
        "是否曾買過主約躉繳投資型" if "是否曾買過主約躉繳投資型" in customer_df.columns else None,
        "是否曾買過躉繳投資型" if "是否曾買過躉繳投資型" in customer_df.columns else None,

        "首次躉繳投資型保單序號" if "首次躉繳投資型保單序號" in customer_df.columns else None,
        "首次躉繳投資型投保日" if "首次躉繳投資型投保日" in customer_df.columns else None,
        "第一張就買躉繳投資型" if "第一張就買躉繳投資型" in customer_df.columns else None,

        "近1年保單數",
        "近1年壽險保單數" if "近1年壽險保單數" in customer_df.columns else None,
        "近1年產險保單數" if "近1年產險保單數" in customer_df.columns else None,
        "近2年保單數",
        "近2年壽險保單數" if "近2年壽險保單數" in customer_df.columns else None,
        "近2年產險保單數" if "近2年產險保單數" in customer_df.columns else None,
        "近3年保單數",
        "近3年壽險保單數" if "近3年壽險保單數" in customer_df.columns else None,
        "近3年產險保單數" if "近3年產險保單數" in customer_df.columns else None,

        "高保費保單數_P75",
        "高保費壽險保單數_P75" if "高保費壽險保單數_P75" in customer_df.columns else None,
        "高保費產險保單數_P75" if "高保費產險保單數_P75" in customer_df.columns else None,
        "高保費保單數_P90",
        "高保費壽險保單數_P90" if "高保費壽險保單數_P90" in customer_df.columns else None,
        "高保費產險保單數_P90" if "高保費產險保單數_P90" in customer_df.columns else None,

        "主約商品險種主類別數" if "主約商品險種主類別數" in customer_df.columns else None,
        "主約商品險種次類別數" if "主約商品險種次類別數" in customer_df.columns else None,
        "主約商品名稱數" if "主約商品名稱數" in customer_df.columns else None,
        "主約繳別類型數" if "主約繳別類型數" in customer_df.columns else None,

        "要被保人同一人比例" if "要被保人同一人比例" in customer_df.columns else None,
        "平均保單間隔天數" if "平均保單間隔天數" in customer_df.columns else None,
        "保單間隔天數中位數" if "保單間隔天數中位數" in customer_df.columns else None,
    ]

    front_cols = [c for c in front_cols if c is not None and c in customer_df.columns]
    other_cols = [c for c in customer_df.columns if c not in front_cols]
    customer_df = customer_df[front_cols + other_cols]

    return customer_df


customer_df = build_customer_table(policy_df)

# print(customer_df.shape)
# print(customer_df.head())
# print(customer_df.columns.tolist())

# %% Build Benchmark
def build_benchmark_snapshot(policy_df: pd.DataFrame, analysis_date=None):
    """
    從 policy-level table 建立 benchmark snapshot

    定義：
    - benchmark 客戶 = 曾買過「躉繳投資型」保單的客戶
    - snapshot = 該客戶「首次買躉繳投資型之前」的所有保單歷史

    參數
    ----------
    policy_df : pd.DataFrame
        每張保單一列的資料表

    analysis_date : str / pd.Timestamp / None
        傳給 build_customer_table() 的分析基準日
        若為 None，預設使用 snapshot 資料中的最大投保日

    回傳
    ----------
    benchmark_first_df : pd.DataFrame
        每位 benchmark 客戶首次買躉繳投資型的資訊

    benchmark_policy_snapshot_df : pd.DataFrame
        benchmark 客戶在首次買躉繳投資型之前的保單層資料

    benchmark_snapshot_df : pd.DataFrame
        由 benchmark_policy_snapshot_df 再聚合而成的客戶層 snapshot 資料
    """

    df = policy_df.copy()
    df.columns = df.columns.str.strip()

    # =========================
    # 0. 檢查必要欄位
    # =========================
    required_cols = [
        "被保人身分證字號",
        "保單申請案號",
        "投保日",
        "保單是否躉繳投資型"
    ]
    missing_required = [c for c in required_cols if c not in df.columns]
    if missing_required:
        raise ValueError(f"policy_df 缺少必要欄位: {missing_required}")

    # =========================
    # 1. 型別整理
    # =========================
    df["投保日"] = pd.to_datetime(df["投保日"], errors="coerce")
    df["保單是否躉繳投資型"] = pd.to_numeric(
        df["保單是否躉繳投資型"], errors="coerce"
    ).fillna(0)

    if "保單序號" in df.columns:
        df["保單序號"] = pd.to_numeric(df["保單序號"], errors="coerce")
    else:
        sort_cols = [c for c in ["被保人身分證字號", "投保日", "保單申請案號"] if c in df.columns]
        df = df.sort_values(sort_cols).reset_index(drop=True)
        df["保單序號"] = df.groupby("被保人身分證字號").cumcount() + 1

    # 若 policy_df 尚未有產壽險拆分欄位，補一版保底
    if "保單件數_壽險" not in df.columns:
        df["保單件數_壽險"] = np.where(df.get("產壽險別", pd.Series(index=df.index)) == "壽險", 1, 0)
    if "保單件數_產險" not in df.columns:
        df["保單件數_產險"] = np.where(df.get("產壽險別", pd.Series(index=df.index)) == "產險", 1, 0)

    if "保單總保費_壽險" not in df.columns:
        df["保單總保費_壽險"] = np.where(
            df.get("產壽險別", pd.Series(index=df.index)) == "壽險",
            df.get("保單總保費", 0),
            0
        )
    if "保單總保費_產險" not in df.columns:
        df["保單總保費_產險"] = np.where(
            df.get("產壽險別", pd.Series(index=df.index)) == "產險",
            df.get("保單總保費", 0),
            0
        )

    if "保單總繳款FYC_壽險" not in df.columns:
        df["保單總繳款FYC_壽險"] = np.where(
            df.get("產壽險別", pd.Series(index=df.index)) == "壽險",
            df.get("保單總繳款FYC", 0),
            0
        )
    if "保單總繳款FYC_產險" not in df.columns:
        df["保單總繳款FYC_產險"] = np.where(
            df.get("產壽險別", pd.Series(index=df.index)) == "產險",
            df.get("保單總繳款FYC", 0),
            0
        )

    # =========================
    # 2. 找出所有 benchmark 保單
    # =========================
    benchmark_policy_all = df[df["保單是否躉繳投資型"] == 1].copy()

    if benchmark_policy_all.empty:
        print("找不到任何『保單是否躉繳投資型 = 1』的資料。")

        benchmark_first_df = pd.DataFrame(columns=[
            "被保人身分證字號",
            "首次躉繳投資型保單申請案號",
            "首次躉繳投資型投保日",
            "首次躉繳投資型保單序號"
        ])

        benchmark_policy_snapshot_df = pd.DataFrame(columns=df.columns.tolist())

        benchmark_snapshot_df = pd.DataFrame(columns=[
            "被保人身分證字號",
            "benchmark標記",
            "首次躉繳投資型保單申請案號",
            "首次躉繳投資型投保日",
            "首次躉繳投資型保單序號"
        ])

        return benchmark_first_df, benchmark_policy_snapshot_df, benchmark_snapshot_df

    # =========================
    # 3. 每位客戶首次買躉繳投資型的資訊
    # =========================
    sort_cols = [c for c in ["被保人身分證字號", "投保日", "保單序號", "保單申請案號"] if c in benchmark_policy_all.columns]
    benchmark_policy_all = benchmark_policy_all.sort_values(sort_cols).copy()

    keep_cols = [
        "被保人身分證字號",
        "保單申請案號",
        "投保日",
        "保單序號"
    ]
    keep_cols = [c for c in keep_cols if c in benchmark_policy_all.columns]

    benchmark_first_df = (
        benchmark_policy_all[keep_cols]
        .drop_duplicates(subset=["被保人身分證字號"], keep="first")
        .rename(columns={
            "保單申請案號": "首次躉繳投資型保單申請案號",
            "投保日": "首次躉繳投資型投保日",
            "保單序號": "首次躉繳投資型保單序號"
        })
        .reset_index(drop=True)
    )

    # =========================
    # 4. 取首次購買之前的保單
    #    規則：投保日 < 首次躉繳投資型投保日
    # =========================
    benchmark_policy_snapshot_df = df.merge(
        benchmark_first_df,
        on="被保人身分證字號",
        how="inner"
    )

    benchmark_policy_snapshot_df = benchmark_policy_snapshot_df[
        benchmark_policy_snapshot_df["投保日"] < benchmark_policy_snapshot_df["首次躉繳投資型投保日"]
    ].copy()

    # =========================
    # 5. 若同一天有多張保單，再用序號更嚴格判斷
    # =========================
    if (
        "保單序號" in benchmark_policy_snapshot_df.columns and
        "首次躉繳投資型保單序號" in benchmark_policy_snapshot_df.columns
    ):
        benchmark_policy_snapshot_df = benchmark_policy_snapshot_df[
            benchmark_policy_snapshot_df["保單序號"] < benchmark_policy_snapshot_df["首次躉繳投資型保單序號"]
        ].copy()

    # =========================
    # 6. snapshot 保單層補充欄位
    # =========================
    if not benchmark_policy_snapshot_df.empty:
        benchmark_policy_snapshot_df["距離首次躉繳投資型天數"] = (
            benchmark_policy_snapshot_df["首次躉繳投資型投保日"] - benchmark_policy_snapshot_df["投保日"]
        ).dt.days

        sort_cols2 = [c for c in ["被保人身分證字號", "投保日", "保單申請案號"] if c in benchmark_policy_snapshot_df.columns]
        benchmark_policy_snapshot_df = benchmark_policy_snapshot_df.sort_values(sort_cols2).reset_index(drop=True)

        benchmark_policy_snapshot_df["snapshot保單序號"] = (
            benchmark_policy_snapshot_df.groupby("被保人身分證字號").cumcount() + 1
        )

        # snapshot 專用：距離首次購買前最近一次投保的 reverse rank
        benchmark_policy_snapshot_df["距離首次購買前倒數第幾張"] = (
            benchmark_policy_snapshot_df.groupby("被保人身分證字號")["snapshot保單序號"]
            .transform(lambda s: s.max() - s + 1)
        )

    # =========================
    # 7. 若沒有任何「購買前保單」，回傳空 snapshot 表
    # =========================
    if benchmark_policy_snapshot_df.empty:
        print("有 benchmark 客戶，但所有人都是第一張就買躉繳投資型，沒有可用的『購買前 snapshot』保單。")

        benchmark_snapshot_df = benchmark_first_df.copy()
        benchmark_snapshot_df["benchmark標記"] = 1

        front_cols = [
            "被保人身分證字號",
            "benchmark標記",
            "首次躉繳投資型保單申請案號",
            "首次躉繳投資型投保日",
            "首次躉繳投資型保單序號",
        ]
        benchmark_snapshot_df = benchmark_snapshot_df[front_cols]

        return benchmark_first_df, benchmark_policy_snapshot_df, benchmark_snapshot_df

    # =========================
    # 8. 用 snapshot 保單層重建 customer-level snapshot
    # =========================
    if analysis_date is None:
        snapshot_analysis_date = benchmark_policy_snapshot_df["投保日"].max()
    else:
        snapshot_analysis_date = pd.to_datetime(analysis_date)

    benchmark_snapshot_df = build_customer_table(
        benchmark_policy_snapshot_df,
        analysis_date=snapshot_analysis_date
    ).copy()

    # =========================
    # 9. merge 回首次購買資訊
    # =========================
    dup_cols = [
        "首次躉繳投資型保單申請案號",
        "首次躉繳投資型投保日",
        "首次躉繳投資型保單序號"
    ]

    benchmark_snapshot_df = benchmark_snapshot_df.drop(
        columns=[c for c in dup_cols if c in benchmark_snapshot_df.columns],
        errors="ignore"
    )

    benchmark_snapshot_df = benchmark_snapshot_df.merge(
        benchmark_first_df,
        on="被保人身分證字號",
        how="left"
    )

    # =========================
    # 10. snapshot 專屬衍生欄位
    # =========================
    if "保單數" in benchmark_snapshot_df.columns:
        benchmark_snapshot_df["首次購買前保單數"] = benchmark_snapshot_df["保單數"]

    if "壽險保單數" in benchmark_snapshot_df.columns:
        benchmark_snapshot_df["首次購買前壽險保單數"] = benchmark_snapshot_df["壽險保單數"]

    if "產險保單數" in benchmark_snapshot_df.columns:
        benchmark_snapshot_df["首次購買前產險保單數"] = benchmark_snapshot_df["產險保單數"]

    if "累計保單總保費" in benchmark_snapshot_df.columns:
        benchmark_snapshot_df["首次購買前累計總保費"] = benchmark_snapshot_df["累計保單總保費"]

    if "累計壽險保單總保費" in benchmark_snapshot_df.columns:
        benchmark_snapshot_df["首次購買前累計壽險總保費"] = benchmark_snapshot_df["累計壽險保單總保費"]

    if "累計產險保單總保費" in benchmark_snapshot_df.columns:
        benchmark_snapshot_df["首次購買前累計產險總保費"] = benchmark_snapshot_df["累計產險保單總保費"]

    if "累計保單總繳款FYC" in benchmark_snapshot_df.columns:
        benchmark_snapshot_df["首次購買前累計總繳款FYC"] = benchmark_snapshot_df["累計保單總繳款FYC"]

    if "累計壽險保單總繳款FYC" in benchmark_snapshot_df.columns:
        benchmark_snapshot_df["首次購買前累計壽險總繳款FYC"] = benchmark_snapshot_df["累計壽險保單總繳款FYC"]

    if "累計產險保單總繳款FYC" in benchmark_snapshot_df.columns:
        benchmark_snapshot_df["首次購買前累計產險總繳款FYC"] = benchmark_snapshot_df["累計產險保單總繳款FYC"]

    if "壽險保單占比" in benchmark_snapshot_df.columns:
        benchmark_snapshot_df["首次購買前壽險保單占比"] = benchmark_snapshot_df["壽險保單占比"]

    if "產險保單占比" in benchmark_snapshot_df.columns:
        benchmark_snapshot_df["首次購買前產險保單占比"] = benchmark_snapshot_df["產險保單占比"]

    if "壽險保費占比" in benchmark_snapshot_df.columns:
        benchmark_snapshot_df["首次購買前壽險保費占比"] = benchmark_snapshot_df["壽險保費占比"]

    if "產險保費占比" in benchmark_snapshot_df.columns:
        benchmark_snapshot_df["首次購買前產險保費占比"] = benchmark_snapshot_df["產險保費占比"]

    if "壽險繳款FYC占比" in benchmark_snapshot_df.columns:
        benchmark_snapshot_df["首次購買前壽險繳款FYC占比"] = benchmark_snapshot_df["壽險繳款FYC占比"]

    if "產險繳款FYC占比" in benchmark_snapshot_df.columns:
        benchmark_snapshot_df["首次購買前產險繳款FYC占比"] = benchmark_snapshot_df["產險繳款FYC占比"]

    if {"首次投保日", "首次躉繳投資型投保日"}.issubset(benchmark_snapshot_df.columns):
        benchmark_snapshot_df["從首次投保到首次買躉繳投資型天數"] = (
            benchmark_snapshot_df["首次躉繳投資型投保日"] - benchmark_snapshot_df["首次投保日"]
        ).dt.days

    if {"最近投保日", "首次躉繳投資型投保日"}.issubset(benchmark_snapshot_df.columns):
        benchmark_snapshot_df["距離首次買躉繳投資型前最近一次投保天數"] = (
            benchmark_snapshot_df["首次躉繳投資型投保日"] - benchmark_snapshot_df["最近投保日"]
        ).dt.days

    # 補上 snapshot 內最後一張保單的資訊
    last_snapshot_policy = (
        benchmark_policy_snapshot_df
        .sort_values(["被保人身分證字號", "投保日", "保單申請案號"])
        .groupby("被保人身分證字號", as_index=False)
        .last()
    )

    last_keep_cols = [
        "被保人身分證字號",
        "保單申請案號",
        "投保日",
        "snapshot保單序號",
        "距離首次躉繳投資型天數",
        "產壽險別",
        "保單件數_壽險",
        "保單件數_產險",
        "保單總保費",
        "保單總保費_壽險",
        "保單總保費_產險",
        "保單總繳款FYC",
        "保單總繳款FYC_壽險",
        "保單總繳款FYC_產險",
        "主約商品名稱",
        "主約商品險種主類別",
        "主約商品險種次類別",
        "主約繳別"
    ]
    last_keep_cols = [c for c in last_keep_cols if c in last_snapshot_policy.columns]

    last_snapshot_policy = last_snapshot_policy[last_keep_cols].rename(columns={
        "保單申請案號": "首次購買前最後一張保單申請案號",
        "投保日": "首次購買前最後一張投保日",
        "snapshot保單序號": "首次購買前最後一張snapshot保單序號",
        "距離首次躉繳投資型天數": "首次購買前最後一張距離天數",
        "產壽險別": "首次購買前最後一張產壽險別",
        "保單件數_壽險": "首次購買前最後一張_壽險件數",
        "保單件數_產險": "首次購買前最後一張_產險件數",
        "保單總保費": "首次購買前最後一張保單總保費",
        "保單總保費_壽險": "首次購買前最後一張壽險保費",
        "保單總保費_產險": "首次購買前最後一張產險保費",
        "保單總繳款FYC": "首次購買前最後一張保單總繳款FYC",
        "保單總繳款FYC_壽險": "首次購買前最後一張壽險繳款FYC",
        "保單總繳款FYC_產險": "首次購買前最後一張產險繳款FYC",
        "主約商品名稱": "首次購買前最後一張主約商品名稱",
        "主約商品險種主類別": "首次購買前最後一張主約商品險種主類別",
        "主約商品險種次類別": "首次購買前最後一張主約商品險種次類別",
        "主約繳別": "首次購買前最後一張主約繳別"
    })

    benchmark_snapshot_df = benchmark_snapshot_df.merge(
        last_snapshot_policy,
        on="被保人身分證字號",
        how="left"
    )

    benchmark_snapshot_df["benchmark標記"] = 1

    # =========================
    # 11. 欄位排序
    # =========================
    front_cols = [
        "被保人身分證字號",
        "benchmark標記",
        "首次躉繳投資型保單申請案號",
        "首次躉繳投資型投保日",
        "首次躉繳投資型保單序號",

        "首次購買前保單數" if "首次購買前保單數" in benchmark_snapshot_df.columns else None,
        "首次購買前壽險保單數" if "首次購買前壽險保單數" in benchmark_snapshot_df.columns else None,
        "首次購買前產險保單數" if "首次購買前產險保單數" in benchmark_snapshot_df.columns else None,

        "首次購買前壽險保單占比" if "首次購買前壽險保單占比" in benchmark_snapshot_df.columns else None,
        "首次購買前產險保單占比" if "首次購買前產險保單占比" in benchmark_snapshot_df.columns else None,
        "首次購買前壽險保費占比" if "首次購買前壽險保費占比" in benchmark_snapshot_df.columns else None,
        "首次購買前產險保費占比" if "首次購買前產險保費占比" in benchmark_snapshot_df.columns else None,
        "首次購買前壽險繳款FYC占比" if "首次購買前壽險繳款FYC占比" in benchmark_snapshot_df.columns else None,
        "首次購買前產險繳款FYC占比" if "首次購買前產險繳款FYC占比" in benchmark_snapshot_df.columns else None,

        "從首次投保到首次買躉繳投資型天數" if "從首次投保到首次買躉繳投資型天數" in benchmark_snapshot_df.columns else None,
        "距離首次買躉繳投資型前最近一次投保天數" if "距離首次買躉繳投資型前最近一次投保天數" in benchmark_snapshot_df.columns else None,

        "首次購買前最後一張保單申請案號" if "首次購買前最後一張保單申請案號" in benchmark_snapshot_df.columns else None,
        "首次購買前最後一張投保日" if "首次購買前最後一張投保日" in benchmark_snapshot_df.columns else None,
        "首次購買前最後一張距離天數" if "首次購買前最後一張距離天數" in benchmark_snapshot_df.columns else None,
        "首次購買前最後一張產壽險別" if "首次購買前最後一張產壽險別" in benchmark_snapshot_df.columns else None,
        "首次購買前最後一張保單總保費" if "首次購買前最後一張保單總保費" in benchmark_snapshot_df.columns else None,
        "首次購買前最後一張壽險保費" if "首次購買前最後一張壽險保費" in benchmark_snapshot_df.columns else None,
        "首次購買前最後一張產險保費" if "首次購買前最後一張產險保費" in benchmark_snapshot_df.columns else None,
    ]
    front_cols = [c for c in front_cols if c is not None and c in benchmark_snapshot_df.columns]
    other_cols = [c for c in benchmark_snapshot_df.columns if c not in front_cols]
    benchmark_snapshot_df = benchmark_snapshot_df[front_cols + other_cols]

    return benchmark_first_df, benchmark_policy_snapshot_df, benchmark_snapshot_df


benchmark_first_df, benchmark_policy_snapshot_df, benchmark_snapshot_df = build_benchmark_snapshot(policy_df)

benchmark_first_df = benchmark_first_df.merge(
    customer_df[["被保人身分證字號", "被保人生日", "被保人性別"]]
    .drop_duplicates(subset=["被保人身分證字號"]),
    on="被保人身分證字號",
    how="left"
)

# %% 躉繳投資型客戶的典型購買路徑
# 第幾張才買躉繳投資型
# 每位 benchmark 客戶首次買躉繳投資型的保單序號分布
seq_dist = (
    benchmark_first_df["首次躉繳投資型保單序號"]
    .value_counts(dropna=False)
    .sort_index()
    .reset_index()
)
seq_dist.columns = ["首次躉繳投資型保單序號", "客戶數"]
seq_dist["比例"] = seq_dist["客戶數"] / seq_dist["客戶數"].sum()

print(seq_dist)


# 分組看：第1張 / 第2張 / 第3張 / 第4張以上
tmp = benchmark_first_df.copy()

tmp["購買路徑分組"] = np.select(
    [
        tmp["首次躉繳投資型保單序號"] == 1,
        tmp["首次躉繳投資型保單序號"] == 2,
        tmp["首次躉繳投資型保單序號"] == 3,
        tmp["首次躉繳投資型保單序號"] >= 4,
    ],
    [
        "第1張就買",
        "第2張才買",
        "第3張才買",
        "第4張以上才買",
    ],
    default="未知"
)

path_group_dist = (
    tmp["購買路徑分組"]
    .value_counts()
    .reset_index()
)
path_group_dist.columns = ["購買路徑分組", "客戶數"]
path_group_dist["比例"] = path_group_dist["客戶數"] / path_group_dist["客戶數"].sum()

print(path_group_dist)

# 買之前有幾張
benchmark_snapshot_df["保單數"].describe()

# 買之前最常見主約
benchmark_policy_snapshot_df["主約商品險種主類別"].value_counts()

pre_policy_dist = (
    benchmark_snapshot_df["保單數"]
    .value_counts()
    .sort_index()
    .reset_index()
)
pre_policy_dist.columns = ["保單數", "客戶數"]
pre_policy_dist["比例"] = pre_policy_dist["客戶數"] / pre_policy_dist["客戶數"].sum()

print(pre_policy_dist)


# 躉繳投資型前一張保單
policy_sorted = policy_df.sort_values(
    ["被保人身分證字號", "投保日", "保單序號"]
).copy()

policy_sorted["前一張商品主類別"] = (
    policy_sorted.groupby("被保人身分證字號")["主約商品險種主類別"].shift(1)
)

policy_sorted["前一張商品名稱"] = (
    policy_sorted.groupby("被保人身分證字號")["主約商品名稱"].shift(1)
)

policy_sorted["前一張保單投保日"] = (
    policy_sorted.groupby("被保人身分證字號")["投保日"].shift(1)
)

# 把 benchmark_first_df 接回來
benchmark_prev_df = benchmark_first_df.merge(
    policy_sorted[
        [
            "被保人身分證字號",
            "保單申請案號",
            "前一張商品主類別",
            "前一張商品名稱",
            "前一張保單投保日"
        ]
    ],
    left_on=["被保人身分證字號", "首次躉繳投資型保單申請案號"],
    right_on=["被保人身分證字號", "保單申請案號"],
    how="left"
)

# 把 benchmark_first_df 接回來
prev_major_dist = (
    benchmark_prev_df["前一張商品主類別"]
    .value_counts(dropna=False)
    .reset_index()
)

prev_major_dist.columns = ["前一張商品主類別", "客戶數"]
prev_major_dist["比例"] = prev_major_dist["客戶數"] / prev_major_dist["客戶數"].sum()

print(prev_major_dist)

# 躉繳前一張與躉繳之間間隔多久
benchmark_prev_df["距離前一張保單天數"] = (
    benchmark_prev_df["首次躉繳投資型投保日"] -
    benchmark_prev_df["前一張保單投保日"]
).dt.days

benchmark_prev_df["距離前一張保單天數"].describe()

# %% Build Candidate Pool
def build_candidate_pool(customer_df):
    df = customer_df.copy()
    df.columns = df.columns.str.strip()

    required_cols = ["被保人身分證字號", "是否曾買過躉繳投資型"]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"customer_df 缺少必要欄位: {missing_cols}")

    df["是否曾買過躉繳投資型"] = pd.to_numeric(df["是否曾買過躉繳投資型"], errors="coerce")

    funnel_df = pd.DataFrame(
        [
            ["全部客戶", len(df)],
            ["是否曾買過躉繳投資型非缺值", df["是否曾買過躉繳投資型"].notna().sum()],
            ["未買躉繳投資型", (df["是否曾買過躉繳投資型"] == 0).sum()],
        ],
        columns=["漏斗階段", "客戶數"]
    )

    candidate_df = df[
        df["是否曾買過躉繳投資型"] == 0
    ].copy()

    candidate_df["candidate標記"] = 1

    return candidate_df, funnel_df

candidate_df, funnel_df = build_candidate_pool(customer_df) # 看漏斗: funnel_df


# %% 建立規則分數 Rule Score 

def build_rule_score_by_segment(
    candidate_df: pd.DataFrame,
    benchmark_snapshot_df: pd.DataFrame,
    benchmark_policy_snapshot_df: pd.DataFrame,
    benchmark_first_df: pd.DataFrame,
    candidate_analysis_date=None,
    gender_col: str = "被保人性別",
    birth_col: str = "被保人生日",
    age_bin_size: int = 10,
    min_benchmark_size: int = 20,
    fallback_to_overall: bool = True
):
    """
    依照 年齡區間 + 性別 分組，建立 benchmark-driven rule score（雙年齡版本）

    年齡定義：
    - candidate：用 被保人生日 + candidate_analysis_date 計算「目前年齡」
    - benchmark：用 被保人生日 + 首次躉繳投資型投保日 計算「購買當下年齡」

    參數
    ----------
    candidate_df : DataFrame
        候選客戶表（未買過躉繳投資型）

    benchmark_snapshot_df : DataFrame
        benchmark 客戶在買前的 customer-level snapshot

    benchmark_policy_snapshot_df : DataFrame
        benchmark 客戶在買前的 policy-level snapshot

    benchmark_first_df : DataFrame
        每位 benchmark 客戶首次買躉繳投資型的資訊
        至少包含：
        - 被保人身分證字號
        - 首次躉繳投資型保單序號
        - 首次躉繳投資型投保日

    candidate_analysis_date : str / Timestamp / None
        candidate 目前年齡的計算基準日
        若為 None，預設用今天

    gender_col : str
        性別欄位名稱，預設 "性別"

    birth_col : str
        生日欄位名稱，預設 "被保人生日"

    age_bin_size : int
        幾歲一個區間，預設 10

    min_benchmark_size : int
        分組 benchmark 低於此人數時，fallback 到 overall benchmark

    fallback_to_overall : bool
        分組樣本不足時是否退回全體 benchmark

    回傳
    ----------
    scored_df : DataFrame
        加上分組 rule score 的 candidate 表

    segment_profile_df : DataFrame
        各 age_gender_group 的 benchmark profile

    segment_product_rate_df : DataFrame
        各 age_gender_group 的商品滲透率表
    """

    # =========================
    # 0. 複製資料
    # =========================
    cand = candidate_df.copy()
    bench_cust = benchmark_snapshot_df.copy()
    bench_pol = benchmark_policy_snapshot_df.copy()
    bench_first = benchmark_first_df.copy()

    for df in [cand, bench_cust, bench_pol, bench_first]:
        df.columns = df.columns.str.strip()

    # =========================
    # 1. 基本檢查
    # =========================
    required_cand_cols = ["被保人身分證字號", birth_col, gender_col, "保單數"]
    required_bench_cust_cols = ["被保人身分證字號", birth_col, gender_col]
    required_bench_first_cols = ["被保人身分證字號", "首次躉繳投資型保單序號", "首次躉繳投資型投保日"]
    required_bench_pol_cols = ["被保人身分證字號", "主約商品險種主類別"]

    missing_cand = [c for c in required_cand_cols if c not in cand.columns]
    missing_bench_cust = [c for c in required_bench_cust_cols if c not in bench_cust.columns]
    missing_bench_first = [c for c in required_bench_first_cols if c not in bench_first.columns]
    missing_bench_pol = [c for c in required_bench_pol_cols if c not in bench_pol.columns]

    if missing_cand:
        raise ValueError(f"candidate_df 缺少必要欄位: {missing_cand}")
    if missing_bench_cust:
        raise ValueError(f"benchmark_snapshot_df 缺少必要欄位: {missing_bench_cust}")
    if missing_bench_first:
        raise ValueError(f"benchmark_first_df 缺少必要欄位: {missing_bench_first}")
    if missing_bench_pol:
        raise ValueError(f"benchmark_policy_snapshot_df 缺少必要欄位: {missing_bench_pol}")

    # =========================
    # 2. 日期 / 數值整理
    # =========================
    if candidate_analysis_date is None:
        candidate_analysis_date = pd.Timestamp.today().normalize()
    else:
        candidate_analysis_date = pd.to_datetime(candidate_analysis_date)

    for df in [cand, bench_cust, bench_first]:
        df[birth_col] = pd.to_datetime(df[birth_col], errors="coerce")

    bench_first["首次躉繳投資型投保日"] = pd.to_datetime(
        bench_first["首次躉繳投資型投保日"], errors="coerce"
    )
    bench_first["首次躉繳投資型保單序號"] = pd.to_numeric(
        bench_first["首次躉繳投資型保單序號"], errors="coerce"
    )

    for df in [cand, bench_cust]:
        if "保單數" in df.columns:
            df["保單數"] = pd.to_numeric(df["保單數"], errors="coerce")

    if "距離最近投保天數" in cand.columns:
        cand["距離最近投保天數"] = pd.to_numeric(cand["距離最近投保天數"], errors="coerce")

    if "距離首次買躉繳投資型前最近一次投保天數" in bench_cust.columns:
        bench_cust["距離首次買躉繳投資型前最近一次投保天數"] = pd.to_numeric(
            bench_cust["距離首次買躉繳投資型前最近一次投保天數"], errors="coerce"
        )

    if "最大單張保單保費" in cand.columns:
        cand["最大單張保單保費"] = pd.to_numeric(cand["最大單張保單保費"], errors="coerce")

    if "最大單張保單保費" in bench_cust.columns:
        bench_cust["最大單張保單保費"] = pd.to_numeric(bench_cust["最大單張保單保費"], errors="coerce")

    # =========================
    # 3. 用生日計算年齡
    # =========================
    def calc_age_years(date_end, date_start):
        return np.floor((date_end - date_start).dt.days / 365.25)

    # candidate：目前年齡
    cand["candidate目前年齡"] = calc_age_years(candidate_analysis_date, cand[birth_col])

    # benchmark：購買當下年齡
    # 先把購買日 merge 到 benchmark_snapshot / benchmark_policy
    # 先刪除 bench_cust 裡可能已存在的同名欄位，避免 merge 後變 _x / _y
    cols_to_drop = ["首次躉繳投資型投保日", "首次躉繳投資型保單序號"]
    bench_cust = bench_cust.drop(columns=[c for c in cols_to_drop if c in bench_cust.columns], errors="ignore")
    
    # 再 merge
    bench_cust = bench_cust.merge(
        bench_first[["被保人身分證字號", "首次躉繳投資型投保日", "首次躉繳投資型保單序號"]],
        on="被保人身分證字號",
        how="left"
    )
    
    # 再計算 benchmark 購買當下年齡
    bench_cust["benchmark購買當下年齡"] = calc_age_years(
        bench_cust["首次躉繳投資型投保日"],
        bench_cust[birth_col]
    )

    bench_first["benchmark購買當下年齡"] = calc_age_years(
        bench_first["首次躉繳投資型投保日"],
        bench_first[birth_col]
    ) if birth_col in bench_first.columns else np.nan

    # 若 benchmark_first 沒有生日，從 bench_cust 補
    if birth_col not in bench_first.columns or bench_first[birth_col].isna().all():
        bench_first = bench_first.merge(
            bench_cust[["被保人身分證字號", birth_col]]
            .drop_duplicates("被保人身分證字號"),
            on="被保人身分證字號",
            how="left"
        )
        bench_first["benchmark購買當下年齡"] = calc_age_years(
            bench_first["首次躉繳投資型投保日"],
            bench_first[birth_col]
        )
    
    # bench_pol 也要有 group
    # 避免 merge 後出現 _x / _y
    bench_pol = bench_pol.drop(
        columns=[c for c in ["benchmark購買當下年齡", gender_col] if c in bench_pol.columns],
        errors="ignore"
    )
    
    bench_pol = bench_pol.merge(
        bench_cust[["被保人身分證字號", "benchmark購買當下年齡", gender_col]]
        .drop_duplicates(subset=["被保人身分證字號"]),
        on="被保人身分證字號",
        how="left"
    )

    # =========================
    # 4. 建立年齡區間 + 性別分組
    # =========================
    def make_age_group(s, bin_size=10):
        floor_age = np.floor(s / bin_size) * bin_size
        floor_age = pd.Series(floor_age, index=s.index)
        lower = floor_age.astype("Int64").astype("string")
        upper = (floor_age + bin_size - 1).astype("Int64").astype("string")
        return lower + "-" + upper

    # candidate 分組：目前年齡
    cand["candidate年齡區間"] = make_age_group(cand["candidate目前年齡"], age_bin_size)
    cand["age_gender_group"] = (
        cand["candidate年齡區間"].fillna("未知") + "_" + cand[gender_col].fillna("未知").astype("string")
    )

    # benchmark 分組：購買當下年齡
    bench_cust["benchmark年齡區間"] = make_age_group(bench_cust["benchmark購買當下年齡"], age_bin_size)
    bench_cust["age_gender_group"] = (
        bench_cust["benchmark年齡區間"].fillna("未知") + "_" + bench_cust[gender_col].fillna("未知").astype("string")
    )

    bench_first["benchmark年齡區間"] = make_age_group(bench_first["benchmark購買當下年齡"], age_bin_size)
    bench_first["age_gender_group"] = (
        bench_first["benchmark年齡區間"].fillna("未知") + "_" + bench_first[gender_col].fillna("未知").astype("string")
    )

    bench_pol["benchmark年齡區間"] = make_age_group(bench_pol["benchmark購買當下年齡"], age_bin_size)
    bench_pol["age_gender_group"] = (
        bench_pol["benchmark年齡區間"].fillna("未知") + "_" + bench_pol[gender_col].fillna("未知").astype("string")
    )

    # =========================
    # 5. 各 segment 的 benchmark 樣本數
    # =========================
    segment_size_df = (
        bench_cust.groupby("age_gender_group")
        .size()
        .reset_index(name="segment_benchmark人數")
    )

    # =========================
    # 6. 保單成熟度：各組第幾張買的分布
    # =========================
    bench_seq = (
        bench_first
        .assign(保單數分箱=lambda x: x["首次躉繳投資型保單序號"].clip(upper=4))
        .groupby(["age_gender_group", "保單數分箱"], dropna=False)
        .size()
        .reset_index(name="人數")
    )
    bench_seq["組內總人數"] = bench_seq.groupby("age_gender_group")["人數"].transform("sum")
    bench_seq["比例"] = bench_seq["人數"] / bench_seq["組內總人數"]

    overall_seq = (
        bench_first
        .assign(保單數分箱=lambda x: x["首次躉繳投資型保單序號"].clip(upper=4))
        .groupby("保單數分箱", dropna=False)
        .size()
        .reset_index(name="人數")
    )
    overall_seq["比例"] = overall_seq["人數"] / overall_seq["人數"].sum()

    seq_map = {
        (row["age_gender_group"], row["保單數分箱"]): row["比例"]
        for _, row in bench_seq.iterrows()
    }
    overall_seq_map = {
        row["保單數分箱"]: row["比例"]
        for _, row in overall_seq.iterrows()
    }

    # =========================
    # 7. 商品路徑：各組 benchmark 商品滲透率
    # =========================
    bench_prod_presence = (
        bench_pol.assign(flag=1)
        .pivot_table(
            index=["被保人身分證字號", "age_gender_group"],
            columns="主約商品險種主類別",
            values="flag",
            aggfunc="max",
            fill_value=0
        )
        .reset_index()
    )

    prod_cols = [c for c in bench_prod_presence.columns if c not in ["被保人身分證字號", "age_gender_group"]]

    segment_product_rate_df = (
        bench_prod_presence.groupby("age_gender_group")[prod_cols]
        .mean()
        .reset_index()
    )

    overall_product_rate = bench_prod_presence[prod_cols].mean() if len(prod_cols) > 0 else pd.Series(dtype=float)
    segment_product_rate_lookup = (
        segment_product_rate_df.set_index("age_gender_group")
        if not segment_product_rate_df.empty else pd.DataFrame()
    )

    # =========================
    # 8. 近期活躍 profile
    # =========================
    if "距離首次買躉繳投資型前最近一次投保天數" in bench_cust.columns:
        active_profile = (
            bench_cust.groupby("age_gender_group")["距離首次買躉繳投資型前最近一次投保天數"]
            .agg(
                active_q25=lambda s: s.quantile(0.25),
                active_q50=lambda s: s.quantile(0.50),
                active_q75=lambda s: s.quantile(0.75),
                benchmark_count="count"
            )
            .reset_index()
        )
        active_profile_lookup = active_profile.set_index("age_gender_group")

        overall_active_profile = {
            "active_q25": bench_cust["距離首次買躉繳投資型前最近一次投保天數"].quantile(0.25),
            "active_q50": bench_cust["距離首次買躉繳投資型前最近一次投保天數"].quantile(0.50),
            "active_q75": bench_cust["距離首次買躉繳投資型前最近一次投保天數"].quantile(0.75),
        }
    else:
        active_profile = pd.DataFrame(columns=["age_gender_group", "active_q25", "active_q50", "active_q75"])
        active_profile_lookup = pd.DataFrame()
        overall_active_profile = {"active_q25": np.nan, "active_q50": np.nan, "active_q75": np.nan}

    # =========================
    # 9. 保費能力 profile
    # =========================
    if "最大單張保單保費" in bench_cust.columns:
        premium_profile = (
            bench_cust.groupby("age_gender_group")["最大單張保單保費"]
            .agg(
                premium_median="median",
                premium_p75=lambda s: s.quantile(0.75),
                benchmark_count="count"
            )
            .reset_index()
        )
        premium_profile_lookup = premium_profile.set_index("age_gender_group")

        overall_premium_profile = {
            "premium_median": bench_cust["最大單張保單保費"].median(),
            "premium_p75": bench_cust["最大單張保單保費"].quantile(0.75),
        }
    else:
        premium_profile = pd.DataFrame(columns=["age_gender_group", "premium_median", "premium_p75"])
        premium_profile_lookup = pd.DataFrame()
        overall_premium_profile = {"premium_median": np.nan, "premium_p75": np.nan}

    # =========================
    # 10. candidate merge 各組樣本數
    # =========================
    scored = cand.merge(segment_size_df, on="age_gender_group", how="left")
    scored["segment_benchmark人數"] = scored["segment_benchmark人數"].fillna(0)

    # =========================
    # 11. rule_保單成熟度
    # =========================
    scored["保單數分箱"] = scored["保單數"].clip(upper=4)

    def get_seq_score(row):
        g = row["age_gender_group"]
        seq = row["保單數分箱"]
        segment_n = row["segment_benchmark人數"]

        if pd.notna(g) and pd.notna(seq):
            if segment_n >= min_benchmark_size and (g, seq) in seq_map:
                return seq_map[(g, seq)]
            elif fallback_to_overall and seq in overall_seq_map:
                return overall_seq_map[seq]
        return np.nan

    scored["rule_保單成熟度"] = scored.apply(get_seq_score, axis=1)

    # =========================
    # 12. rule_商品路徑
    # =========================
    candidate_product_flag_cols = [
        "是否曾買過壽險",
        "是否曾買過健康險",
        "是否曾買過傷害險",
        "是否曾買過投資型",
        "是否曾買過年金險",
    ]
    product_name_map = {
        "是否曾買過壽險": "壽險",
        "是否曾買過健康險": "健康險",
        "是否曾買過傷害險": "傷害險",
        "是否曾買過投資型": "投資型",
        "是否曾買過年金險": "年金險",
    }

    for c in candidate_product_flag_cols:
        if c in scored.columns:
            scored[c] = pd.to_numeric(scored[c], errors="coerce").fillna(0)

    def get_product_score(row):
        g = row["age_gender_group"]
        segment_n = row["segment_benchmark人數"]

        score = 0.0
        weight_sum = 0.0

        for flag_col, prod_name in product_name_map.items():
            if flag_col not in row.index:
                continue

            if segment_n >= min_benchmark_size and not segment_product_rate_lookup.empty and g in segment_product_rate_lookup.index:
                w = segment_product_rate_lookup.loc[g].get(prod_name, np.nan)
            elif fallback_to_overall:
                w = overall_product_rate.get(prod_name, np.nan)
            else:
                w = np.nan

            if pd.notna(w):
                score += row[flag_col] * w
                weight_sum += w

        if weight_sum > 0:
            return score / weight_sum
        return np.nan

    scored["rule_商品路徑"] = scored.apply(get_product_score, axis=1)

    # =========================
    # 13. rule_近期活躍
    # =========================
    def get_active_score(row):
        if "距離最近投保天數" not in row.index:
            return np.nan

        x = row["距離最近投保天數"]
        g = row["age_gender_group"]
        segment_n = row["segment_benchmark人數"]

        if pd.isna(x):
            return np.nan

        if segment_n >= min_benchmark_size and not active_profile_lookup.empty and g in active_profile_lookup.index:
            q25 = active_profile_lookup.loc[g, "active_q25"]
            q50 = active_profile_lookup.loc[g, "active_q50"]
            q75 = active_profile_lookup.loc[g, "active_q75"]
        elif fallback_to_overall:
            q25 = overall_active_profile["active_q25"]
            q50 = overall_active_profile["active_q50"]
            q75 = overall_active_profile["active_q75"]
        else:
            return np.nan

        if pd.isna(q25) or pd.isna(q50) or pd.isna(q75):
            return np.nan

        if x <= q25:
            return 1.0
        elif x <= q50:
            return 0.8
        elif x <= q75:
            return 0.5
        else:
            return 0.2

    scored["rule_近期活躍"] = scored.apply(get_active_score, axis=1)

    # =========================
    # 14. rule_保費能力
    # =========================
    def get_premium_score(row):
        if "最大單張保單保費" not in row.index:
            return np.nan

        x = row["最大單張保單保費"]
        g = row["age_gender_group"]
        segment_n = row["segment_benchmark人數"]

        if pd.isna(x):
            return np.nan

        if segment_n >= min_benchmark_size and not premium_profile_lookup.empty and g in premium_profile_lookup.index:
            med = premium_profile_lookup.loc[g, "premium_median"]
            p75 = premium_profile_lookup.loc[g, "premium_p75"]
        elif fallback_to_overall:
            med = overall_premium_profile["premium_median"]
            p75 = overall_premium_profile["premium_p75"]
        else:
            return np.nan

        if pd.isna(med) or pd.isna(p75):
            return np.nan

        if x >= p75:
            return 1.0
        elif x >= med:
            return 0.7
        else:
            return 0.4

    scored["rule_保費能力"] = scored.apply(get_premium_score, axis=1)

    # =========================
    # 15. normalize
    # =========================
    rule_cols = [
        "rule_保單成熟度",
        "rule_商品路徑",
        "rule_近期活躍",
        "rule_保費能力",
    ]

    for c in rule_cols:
        scored[c + "_norm"] = scored[c].rank(pct=True)

    # =========================
    # 16. 最終 rule_score
    # =========================
    weight_map = {
        "rule_保單成熟度_norm": 0.35,
        "rule_商品路徑_norm": 0.30,
        "rule_近期活躍_norm": 0.20,
        "rule_保費能力_norm": 0.15,
    }

    numerator = pd.Series(0.0, index=scored.index)
    denominator = pd.Series(0.0, index=scored.index)

    for col, w in weight_map.items():
        if col in scored.columns:
            has_value = scored[col].notna().astype(float)
            numerator += scored[col].fillna(0) * w
            denominator += has_value * w

    scored["rule_score"] = np.where(
        denominator > 0,
        numerator / denominator,
        np.nan
    )
    scored["rule_score_pct"] = scored["rule_score"].rank(pct=True)

    # =========================
    # 17. 排名
    # =========================
    scored["segment_rule_rank"] = (
        scored.groupby("age_gender_group")["rule_score"]
        .rank(method="dense", ascending=False)
    )

    sort_cols = ["rule_score"]
    ascending = [False]

    if "保單數" in scored.columns:
        sort_cols.append("保單數")
        ascending.append(False)

    if "最大單張保單保費" in scored.columns:
        sort_cols.append("最大單張保單保費")
        ascending.append(False)

    if "距離最近投保天數" in scored.columns:
        sort_cols.append("距離最近投保天數")
        ascending.append(True)

    scored = scored.sort_values(sort_cols, ascending=ascending).reset_index(drop=True)
    scored["overall_rule_rank"] = np.arange(1, len(scored) + 1)

    # =========================
    # 18. 輸出 profile
    # =========================
    segment_profile_df = segment_size_df.copy()

    if not active_profile.empty:
        segment_profile_df = segment_profile_df.merge(
            active_profile.drop(columns=["benchmark_count"], errors="ignore"),
            on="age_gender_group",
            how="left"
        )

    if not premium_profile.empty:
        segment_profile_df = segment_profile_df.merge(
            premium_profile.drop(columns=["benchmark_count"], errors="ignore"),
            on="age_gender_group",
            how="left"
        )

    segment_profile_df = segment_profile_df.sort_values("segment_benchmark人數", ascending=False)

    # =========================
    # 19. 欄位順序
    # =========================
    front_cols = [
        "被保人身分證字號",
        birth_col if birth_col in scored.columns else None,
        gender_col if gender_col in scored.columns else None,
        "candidate目前年齡",
        "candidate年齡區間",
        "age_gender_group",
        "segment_benchmark人數",
        "保單數" if "保單數" in scored.columns else None,
        "距離最近投保天數" if "距離最近投保天數" in scored.columns else None,
        "最大單張保單保費" if "最大單張保單保費" in scored.columns else None,
        "rule_保單成熟度",
        "rule_商品路徑",
        "rule_近期活躍",
        "rule_保費能力",
        "rule_score",
        "rule_score_pct",
        "segment_rule_rank",
        "overall_rule_rank",
    ]
    front_cols = [c for c in front_cols if c is not None and c in scored.columns]
    other_cols = [c for c in scored.columns if c not in front_cols]
    scored = scored[front_cols + other_cols]

    return scored, segment_profile_df, segment_product_rate_df


seg_rule_scored_df, segment_profile_df, segment_product_rate_df = build_rule_score_by_segment(
    candidate_df=candidate_df,
    benchmark_snapshot_df=benchmark_snapshot_df,
    benchmark_policy_snapshot_df=benchmark_policy_snapshot_df,
    benchmark_first_df=benchmark_first_df,
    age_col="被保人目前年齡",
    gender_col="被保人性別",
    age_bin_size=10,
    min_benchmark_size=20,
    fallback_to_overall=True
)

# benchmark 分組人數
segment_profile_df.head(20)

# 看某個 segment 的前幾名
seg_rule_scored_df[
    seg_rule_scored_df["age_gender_group"] == "40-49_女"
].head(20)



# %% 買躉繳投資型前的前三張保單路徑排名

def build_top3_route_rank(benchmark_policy_snapshot_df, top_n=50):
    """
    計算 benchmark 客戶在首次買躉繳投資型前的『前三張保單路徑排名』

    參數
    ----------
    benchmark_policy_snapshot_df : DataFrame
        benchmark 客戶在首次買躉繳投資型之前的保單層資料
        必要欄位：
        - 被保人身分證字號
        - 投保日
        - 保單序號
        - 主約商品險種主類別

    top_n : int
        顯示前幾名路徑

    回傳
    ----------
    route_detail_df : DataFrame
        每位客戶對應的前三張路徑明細

    route_rank_df : DataFrame
        路徑排名表
    """

    df = benchmark_policy_snapshot_df.copy()
    df.columns = df.columns.str.strip()

    required_cols = [
        "被保人身分證字號",
        "投保日",
        "保單序號",
        "主約商品險種主類別"
    ]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"benchmark_policy_snapshot_df 缺少必要欄位: {missing_cols}")

    df["投保日"] = pd.to_datetime(df["投保日"], errors="coerce")
    df["保單序號"] = pd.to_numeric(df["保單序號"], errors="coerce")

    # 依客戶排序
    df = df.sort_values(["被保人身分證字號", "投保日", "保單序號"]).copy()

    # 每位客戶只取『買前最後 3 張』
    df["買前倒數序號"] = df.groupby("被保人身分證字號").cumcount(ascending=True) + 1
    df["買前總保單數"] = df.groupby("被保人身分證字號")["被保人身分證字號"].transform("count")
    df["距離最後一張的排序"] = df["買前總保單數"] - df["買前倒數序號"] + 1

    # 只保留最後3張
    last3 = df[df["距離最後一張的排序"] <= 3].copy()

    # 為了組路徑，要重新標號為 第1張/第2張/第3張（從早到晚）
    last3 = last3.sort_values(["被保人身分證字號", "投保日", "保單序號"]).copy()
    last3["前三張內順序"] = last3.groupby("被保人身分證字號").cumcount() + 1

    # pivot 成一人一列
    route_detail_df = (
        last3.pivot_table(
            index="被保人身分證字號",
            columns="前三張內順序",
            values="主約商品險種主類別",
            aggfunc="first"
        )
        .reset_index()
    )

    # 欄位改名
    rename_map = {
        1: "前第1張商品",
        2: "前第2張商品",
        3: "前第3張商品",
    }
    route_detail_df = route_detail_df.rename(columns=rename_map)

    # 不足3張的補 Start
    for c in ["前第1張商品", "前第2張商品", "前第3張商品"]:
        if c not in route_detail_df.columns:
            route_detail_df[c] = np.nan
        route_detail_df[c] = route_detail_df[c].fillna("Start")

    # 組成完整路徑
    route_detail_df["前三張保單路徑"] = (
        route_detail_df["前第1張商品"] + " → " +
        route_detail_df["前第2張商品"] + " → " +
        route_detail_df["前第3張商品"] + " → 躉繳投資型"
    )

    # 排名
    route_rank_df = (
        route_detail_df["前三張保單路徑"]
        .value_counts()
        .reset_index()
    )
    route_rank_df.columns = ["前三張保單路徑", "客戶數"]
    route_rank_df["比例"] = route_rank_df["客戶數"] / route_rank_df["客戶數"].sum()
    route_rank_df = route_rank_df.head(top_n).copy()

    return route_detail_df, route_rank_df


route_detail_df, route_rank_df = build_top3_route_rank(
    benchmark_policy_snapshot_df=benchmark_policy_snapshot_df,
    top_n=50
)

# print(route_rank_df.head(20))


# %% 倒數位置商品分布

def build_pre_purchase_position_dist(benchmark_policy_snapshot_df):
    df = benchmark_policy_snapshot_df.copy()
    df.columns = df.columns.str.strip()

    required_cols = [
        "被保人身分證字號",
        "投保日",
        "保單序號",
        "主約商品險種主類別"
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"缺少必要欄位: {missing}")

    df["投保日"] = pd.to_datetime(df["投保日"], errors="coerce")
    df["保單序號"] = pd.to_numeric(df["保單序號"], errors="coerce")

    df = df.sort_values(["被保人身分證字號", "投保日", "保單序號"]).copy()

    # 買前總保單數
    df["買前總保單數"] = df.groupby("被保人身分證字號")["被保人身分證字號"].transform("count")
    df["買前順序"] = df.groupby("被保人身分證字號").cumcount() + 1

    # 距離躉繳投資型前的位置：1 = 前一張，2 = 前二張 ...
    df["距離躉繳投資型前的位置"] = df["買前總保單數"] - df["買前順序"] + 1

    long_df = df[[
        "被保人身分證字號",
        "投保日",
        "保單序號",
        "主約商品險種主類別",
        "距離躉繳投資型前的位置"
    ]].copy()

    dist_df = (
        long_df.groupby(["距離躉繳投資型前的位置", "主約商品險種主類別"], dropna=False)
        .size()
        .reset_index(name="客戶數")
    )
    dist_df["位置總數"] = dist_df.groupby("距離躉繳投資型前的位置")["客戶數"].transform("sum")
    dist_df["比例"] = dist_df["客戶數"] / dist_df["位置總數"]

    return long_df, dist_df

pre_position_long_df, pre_position_dist_df = build_pre_purchase_position_dist(
    benchmark_policy_snapshot_df
)

# %% 買前每一張保單購買間距

def build_pre_purchase_gap_long(benchmark_policy_snapshot_df, benchmark_first_df):
    df = benchmark_policy_snapshot_df.copy()
    first_df = benchmark_first_df.copy()

    df.columns = df.columns.str.strip()
    first_df.columns = first_df.columns.str.strip()

    required_df_cols = [
        "被保人身分證字號",
        "投保日",
        "保單序號",
        "主約商品險種主類別"
    ]
    required_first_cols = [
        "被保人身分證字號",
        "首次躉繳投資型投保日",
        "首次躉繳投資型保單序號"
    ]

    missing_df = [c for c in required_df_cols if c not in df.columns]
    missing_first = [c for c in required_first_cols if c not in first_df.columns]

    if missing_df:
        raise ValueError(f"benchmark_policy_snapshot_df 缺少必要欄位: {missing_df}")
    if missing_first:
        raise ValueError(f"benchmark_first_df 缺少必要欄位: {missing_first}")

    df["投保日"] = pd.to_datetime(df["投保日"], errors="coerce")
    df["保單序號"] = pd.to_numeric(df["保單序號"], errors="coerce")

    first_df["首次躉繳投資型投保日"] = pd.to_datetime(first_df["首次躉繳投資型投保日"], errors="coerce")
    first_df["首次躉繳投資型保單序號"] = pd.to_numeric(first_df["首次躉繳投資型保單序號"], errors="coerce")

    df = df.sort_values(["被保人身分證字號", "投保日", "保單序號"]).copy()

    # merge 首次躉繳投資型資訊
    merge_cols = ["被保人身分證字號", "首次躉繳投資型投保日", "首次躉繳投資型保單序號"]
    for c in merge_cols[1:]:
        if c in df.columns:
            df = df.drop(columns=c)

    df = df.merge(first_df[merge_cols], on="被保人身分證字號", how="left")

    # 買前順序 / 倒數位置
    df["買前總保單數"] = df.groupby("被保人身分證字號")["被保人身分證字號"].transform("count")
    df["買前順序"] = df.groupby("被保人身分證字號").cumcount() + 1
    df["距離躉繳投資型前的位置"] = df["買前總保單數"] - df["買前順序"] + 1

    # 與前一張保單間距
    df["前一張投保日"] = df.groupby("被保人身分證字號")["投保日"].shift(1)
    df["與前一張保單間距天數"] = (df["投保日"] - df["前一張投保日"]).dt.days

    # 距離首次躉繳投資型天數
    df["距離首次躉繳投資型天數"] = (
        df["首次躉繳投資型投保日"] - df["投保日"]
    ).dt.days

    # 與下一張買前保單間距（有時也會想看）
    df["下一張投保日"] = df.groupby("被保人身分證字號")["投保日"].shift(-1)
    df["與下一張買前保單間距天數"] = (df["下一張投保日"] - df["投保日"]).dt.days

    long_df = df[[
        "被保人身分證字號",
        "投保日",
        "保單序號",
        "主約商品險種主類別",
        "買前順序",
        "距離躉繳投資型前的位置",
        "與前一張保單間距天數",
        "與下一張買前保單間距天數",
        "距離首次躉繳投資型天數",
        "首次躉繳投資型投保日",
        "首次躉繳投資型保單序號"
    ]].copy()
    
    # =========================
    # 補上：買躉繳投資型前前三個購買商品
    # 目的：讓 Tableau 可以直接用這個欄位做路徑排名與客戶明細連動
    # =========================
    
    # 只取買前最後三張保單
    top3_df = long_df[long_df["距離躉繳投資型前的位置"] <= 3].copy()
    
    # 依照「從早到晚」排序，避免路徑順序顛倒
    top3_df = top3_df.sort_values(
        ["被保人身分證字號", "投保日", "保單序號"]
    ).copy()
    
    # 在前三張內重新編順序：1,2,3（從早到晚）
    top3_df["前三張內順序"] = top3_df.groupby("被保人身分證字號").cumcount() + 1
    
    # pivot 成一人一列
    top3_route_df = (
        top3_df.pivot_table(
            index="被保人身分證字號",
            columns="前三張內順序",
            values="主約商品險種主類別",
            aggfunc="first"
        )
        .reset_index()
    )
    
    # 欄位改名
    top3_route_df = top3_route_df.rename(columns={
        1: "前第1張商品",
        2: "前第2張商品",
        3: "前第3張商品",
    })
    
    # 不足三張的補 Start
    for c in ["前第1張商品", "前第2張商品", "前第3張商品"]:
        if c not in top3_route_df.columns:
            top3_route_df[c] = np.nan
        top3_route_df[c] = top3_route_df[c].fillna("Start")
    
    # 組合成完整路徑字串
    top3_route_df["買躉繳投資型前前三個購買商品"] = (
        top3_route_df["前第1張商品"] + " → " +
        top3_route_df["前第2張商品"] + " → " +
        top3_route_df["前第3張商品"] + " → 躉繳投資型"
    )
    
    # merge 回 long_df，讓同一位客戶的每一列保單都帶著同一條路徑
    long_df = long_df.merge(
        top3_route_df[["被保人身分證字號", "買躉繳投資型前前三個購買商品"]],
        on="被保人身分證字號",
        how="left"
    )

    return long_df

pre_purchase_gap_long_df = build_pre_purchase_gap_long(
    benchmark_policy_snapshot_df,
    benchmark_first_df
)

# %% output

pre_position_long_df.to_excel("D:/投資型/lump/tableau/pre_position_long_df.xlsx", index = False)
pre_position_dist_df.to_excel("D:/投資型/lump/tableau/pre_position_dist_df.xlsx", index = False)
pre_purchase_gap_long_df.to_excel("D:/投資型/lump/tableau/pre_purchase_gap_long_df.xlsx", index = False)
route_rank_df.to_excel("D:/投資型/lump/tableau/route_rank_df.xlsx", index = False)
route_detail_df.to_excel("D:/投資型/lump/tableau/route_detail_df.xlsx", index = False)
