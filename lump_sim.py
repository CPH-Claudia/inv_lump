# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 09:03:51 2026

@author: Z01788
"""

import pantab
import pandas as pd
import numpy as np


NEEDED_COLS = [
    # 核心ID
    "保單申請案號", "被保人身分證字號", "要保人身分證字號",

    # 日期
    "投保日", "生日", "被保人生日",

    # 客戶屬性
    "性別", "被保人性別", "婚姻", "學歷", "被保人目前年齡", 
    "(要)保人-家庭年收入(萬)", "(要)保人-工作年收入(萬)", "(要)保人-其他年收入(萬)", 

    # 商品 / 保單結構
    "商品名稱", "商品系統代碼", "商品險種主類別", "商品險種次類別",
    "型別/計劃別", "主附約別", "繳別", "保單狀況", "產壽險別",
    "繳費期間(起)", "繳費期間(迄)"

    # 金額
    "繳款保費", "繳款FYC",

    # 組織 / 歸屬
    "經紀人1", "經紀人1業代", "營業單位", "保險公司", "保險公司代碼"
]

hyper_path = r"D:\投資型\lump\ipo_0306.hyper"
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



def build_policy_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    將保單明細層資料整理成 policy-level table（一張保單一列）

    回傳：
    - policy_df：每張保單一列
    """

    df = df.copy()

    # # =========================
    # # 0. 檢查必要欄位
    # # =========================
    # required_cols = [
    #     "保單申請案號",
    #     "被保人身分證號",
    #     "要保人身分證號",
    #     "投保日",
    #     "商品名稱",
    #     "商品險種主類別",
    #     "商品險種次類別",
    #     "主附約別",
    #     "繳別",
    #     "繳款保費",
    #     "新繳款fyc",
    #     "保單狀況",
    #     "業務員",
    #     "營業單位",
    # ]
    # missing_required = [c for c in required_cols if c not in df.columns]
    # if missing_required:
    #     raise ValueError(f"缺少必要欄位: {missing_required}")

    # =========================
    # 1. 基本清理
    # =========================
    # 欄位名稱去空白
    df.columns = df.columns.str.strip()

    # 常見字串欄位去空白
    str_cols = [
        "保單申請案號", "被保人身分證字號", "要保人身分證字號",
        "商品名稱", "商品系統代碼", "商品險種主類別", "商品險種次類別",
        "型別/計劃別", "主附約別", "繳別", "保單狀況", 
        "經紀人1", "經紀人1業代", "營業單位", "產壽險別", "保險公司", "保險公司代碼",
        "性別", "被保人性別", "婚姻", "學歷"
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
    date_cols = ["投保日", "生日", "被保人生日", "繳費期間(起)", "繳費期間(迄)"]
    for c in date_cols:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")

    # 數值欄位
    num_cols = ["繳款保費", "繳款FYC", "被保人目前年齡", "要保人年收入", "繳費年期"]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # 排除沒有保單申請案號的資料
    df = df[df["保單申請案號"].notna()].copy()

    # =========================
    # 2. 建立明細層 flags
    # =========================
    df["是否主約"] = (df["主附約別"] == "主約").astype("Int64")
    df["是否附約"] = (df["主附約別"] == "附約").astype("Int64")
    # df["是否投資型明細"] = (df["商品險種主類別"] == "投資型").astype("Int64")
    # df["是否躉繳明細"] = (df["繳別"] == "躉繳").astype("Int64")
    df["是否躉繳投資型"] = (
        (df["商品險種主類別"] == "投資型") &
        (df["繳別"] == "躉繳")
    ).astype("Int64")

    df["要被保人是否同一人"] = (
        df["被保人身分證字號"].fillna("") == df["要保人身分證字號"].fillna("")
    ).astype("Int64")

    # =========================
    # 3. 找主約資料
    #    若同保單有多筆主約，保留排序後第一筆
    # =========================
    # 排序邏輯：
    # 1. 是否主約優先
    # 2. 投保日早的優先
    # 3. 保費高的優先（避免異常重複時抓到很小筆）
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

    主約欄位 = [
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
    主約欄位 = [c for c in 主約欄位 if c in main_df.columns]

    main_df = (
        main_df[主約欄位]
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
    # 先定義 groupby 聚合
    agg_dict = {
        "被保人身分證字號": "first",
        "要保人身分證字號": "first",
        "投保日": "min",
        "保單狀況": "first",
        "經紀人1業代": "first",
        "營業單位": "first",

        # "保單申請案號": "size",      # 明細列數
        "是否主約": "sum",
        "是否附約": "sum",
        "繳款保費": "sum",
        "繳款FYC": "sum",

        "商品名稱": pd.Series.nunique,
        "商品險種主類別": pd.Series.nunique,
        "商品險種次類別": pd.Series.nunique,

        # "是否投資型明細": "max",
        # "是否躉繳明細": "max",
        "是否躉繳投資型": "max",
        "要被保人是否同一人": "max",
    }

    # 選配欄位
    optional_first_cols = [
        "產壽險別", "保險公司", "保險公司代碼",
        "繳費年期", "繳費期間(起)", "繳費期間(迄)",
        "性別", "婚姻", "學歷", "被保人投保年齡", "被保人目前年齡",
        "要保人年收入", "生日"
    ]
    for c in optional_first_cols:
        if c in df.columns:
            agg_dict[c] = "first"

    policy_df = (
        df.groupby("保單申請案號", dropna=False)
        .agg(agg_dict)
        .reset_index()
        .rename(columns={
            # "保單申請案號": "保單申請案號",
            "被保人身分證字號": "被保人身分證字號",
            "要保人身分證字號": "要保人身分證字號",
            "投保日": "投保日",
            "保單狀況": "保單狀況",
            "經紀人1業代": "經紀人1業代",
            "營業單位": "營業單位",

            "繳款保費": "保單總保費",
            "繳款FYC": "保單總繳款FYC",
            "商品名稱": "商品名稱數",
            "商品險種主類別": "商品險種主類別數",
            "商品險種次類別": "商品險種次類別數",

            "是否主約": "主約筆數",
            "是否附約": "附約筆數",
            # "是否投資型明細": "是否含投資型",
            # "是否躉繳明細": "是否含躉繳",
            "是否躉繳投資型": "是否含躉繳投資型",
            "要被保人是否同一人": "要被保人是否同一人"
        })
    )
    
    # 再另外補「明細列數」
    detail_cnt = (
        df.groupby("保單申請案號", dropna=False)
        .size()
        .reset_index(name="明細列數")
    )
    
    policy_df = policy_df.merge(detail_cnt, on="保單申請案號", how="left")

    # 明細列數改名
    policy_df = policy_df.rename(columns={"保單申請案號": "保單申請案號_tmp"})
    policy_df["明細列數"] = policy_df["保單申請案號_tmp"]
    policy_df = policy_df.rename(columns={"保單申請案號_tmp": "保單申請案號"})

    # 上面 groupby size 那欄因為也叫保單申請案號，這邊修正成真正欄位名
    if "保單申請案號" in policy_df.columns and "明細列數" not in policy_df.columns:
        pass

    # 若明細列數被處理錯，重新補
    if "明細列數" not in policy_df.columns or policy_df["明細列數"].dtype == "object":
        detail_cnt = (
            df.groupby("保單申請案號", dropna=False)
            .size()
            .reset_index(name="明細列數")
        )
        policy_df = policy_df.drop(columns=["明細列數"], errors="ignore").merge(
            detail_cnt,
            on="保單申請案號",
            how="left"
        )

    # =========================
    # 5. merge 主約資訊
    # =========================
    policy_df = policy_df.merge(
        main_df,
        on="保單申請案號",
        how="left"
    )

    # =========================
    # 6. 建立保單層判斷欄位
    # =========================
    # policy_df["主約是否投資型"] = (
    #     policy_df["主約商品險種主類別"] == "投資型"
    # ).astype("Int64")

    # policy_df["主約是否躉繳"] = (
    #     policy_df["主約繳別"] == "躉繳"
    # ).astype("Int64")

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
    
    policy_df["保單總保費"] = pd.to_numeric(policy_df["保單總保費"], errors="coerce")
    policy_df["保單總繳款FYC"] = pd.to_numeric(policy_df["保單總繳款FYC"], errors="coerce")
    policy_df["明細列數"] = pd.to_numeric(policy_df["明細列數"], errors="coerce")
    
    policy_df["平均每明細保費"] = policy_df["保單總保費"] / policy_df["明細列數"]
    policy_df["平均每明細繳款FYC"] = policy_df["保單總繳款FYC"] / policy_df["明細列數"]

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
        "保單總保費",
        "保單總繳款FYC",

        # "是否含投資型",
        # "是否含躉繳",
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

print(policy_df.shape)
print(policy_df.head())
print(policy_df.columns.tolist())




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

    # 日期欄位
    if "投保日" in df.columns:
        df["投保日"] = pd.to_datetime(df["投保日"], errors="coerce")
    if "生日" in df.columns:
        df["生日"] = pd.to_datetime(df["生日"], errors="coerce")

    # 數值欄位
    num_cols = [
        "保單序號",
        "主約保費",
        "主約新繳款FYC" if "主約新繳款FYC" in df.columns else None,
        "保單總保費",
        "保單總繳款FYC",
        "是否含投資型",
        "是否含躉繳",
        "是否含躉繳投資型",
        "主約是否投資型",
        "主約是否躉繳",
        "主約是否躉繳投資型",
        "保單是否躉繳投資型",
        "要被保人是否同一人",
        "是否有附約" if "是否有附約" in df.columns else None,
        "被保人投保年齡" if "被保人投保年齡" in df.columns else None,
        "被保人目前年齡" if "被保人目前年齡" in df.columns else None,
        "要保人年收入" if "要保人年收入" in df.columns else None,
    ]
    num_cols = [c for c in num_cols if c is not None and c in df.columns]

    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # 排序
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
    # 4. 時間旗標
    # =========================
    df["近1年保單"] = (df["投保日"] >= (analysis_date - pd.Timedelta(days=365))).astype("Int64")
    df["近2年保單"] = (df["投保日"] >= (analysis_date - pd.Timedelta(days=730))).astype("Int64")
    df["近3年保單"] = (df["投保日"] >= (analysis_date - pd.Timedelta(days=1095))).astype("Int64")

    # =========================
    # 5. 高保費旗標（以全體 P75 當第一版門檻）
    # =========================
    premium_base = df["保單總保費"].dropna()
    if len(premium_base) > 0:
        p75 = premium_base.quantile(0.75)
        p90 = premium_base.quantile(0.90)
    else:
        p75 = np.nan
        p90 = np.nan

    df["是否高保費保單_P75"] = (df["保單總保費"] >= p75).astype("Int64") if pd.notna(p75) else 0
    df["是否高保費保單_P90"] = (df["保單總保費"] >= p90).astype("Int64") if pd.notna(p90) else 0

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
        "保單總保費": ["sum", "mean", "median", "max"],
        "保單總繳款FYC": ["sum", "mean", "max"],
        "主約保費": ["mean", "median", "max"] if "主約保費" in df.columns else ["count"],

        "是否含投資型": "sum" if "是否含投資型" in df.columns else "count",
        "是否含躉繳": "sum" if "是否含躉繳" in df.columns else "count",
        "是否含躉繳投資型": "sum" if "是否含躉繳投資型" in df.columns else "count",

        "主約是否投資型": "sum" if "主約是否投資型" in df.columns else "count",
        "主約是否躉繳": "sum" if "主約是否躉繳" in df.columns else "count",
        "主約是否躉繳投資型": "sum" if "主約是否躉繳投資型" in df.columns else "count",
        "保單是否躉繳投資型": "sum" if "保單是否躉繳投資型" in df.columns else "count",

        "近1年保單": "sum",
        "近2年保單": "sum",
        "近3年保單": "sum",

        "是否高保費保單_P75": "sum",
        "是否高保費保單_P90": "sum",

        "主約_壽險": "sum",
        "主約_健康險": "sum",
        "主約_傷害險": "sum",
        "主約_投資型": "sum",
        "主約_年金險": "sum",

        "保單間隔天數": ["mean", "median", "min", "max"],
        "要被保人是否同一人": "mean" if "要被保人是否同一人" in df.columns else "count",
    }

    # 動態加入可選欄位
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
    if "性別" in df.columns:
        agg_dict["性別"] = mode_or_nan
    if "婚姻" in df.columns:
        agg_dict["婚姻"] = mode_or_nan
    if "學歷" in df.columns:
        agg_dict["學歷"] = mode_or_nan
    if "被保人投保年齡" in df.columns:
        agg_dict["被保人投保年齡"] = ["min", "max", "mean"]
    if "被保人目前年齡" in df.columns:
        agg_dict["被保人目前年齡"] = last_valid
    if "要保人年收入" in df.columns:
        agg_dict["要保人年收入"] = "max"
    if "生日" in df.columns:
        agg_dict["生日"] = "first"

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

        "保單總保費_sum": "累計保單總保費",
        "保單總保費_mean": "平均每張保單保費",
        "保單總保費_median": "保單保費中位數",
        "保單總保費_max": "最大單張保單保費",

        "保單總繳款FYC_sum": "累計保單總繳款FYC",
        "保單總繳款FYC_mean": "平均每張保單繳款FYC",
        "保單總繳款FYC_max": "最大單張保單繳款FYC",

        "主約保費_mean": "平均主約保費",
        "主約保費_median": "主約保費中位數",
        "主約保費_max": "最大主約保費",

        "是否含投資型_sum": "含投資型保單數",
        "是否含躉繳_sum": "含躉繳保單數",
        "是否含躉繳投資型_sum": "含躉繳投資型保單數",

        "主約是否投資型_sum": "主約投資型保單數",
        "主約是否躉繳_sum": "主約躉繳保單數",
        "主約是否躉繳投資型_sum": "主約躉繳投資型保單數",
        "保單是否躉繳投資型_sum": "躉繳投資型保單數",

        "近1年保單_sum": "近1年保單數",
        "近2年保單_sum": "近2年保單數",
        "近3年保單_sum": "近3年保單數",

        "是否高保費保單_P75_sum": "高保費保單數_P75",
        "是否高保費保單_P90_sum": "高保費保單數_P90",

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
        "性別_mode_or_nan": "性別",
        "婚姻_mode_or_nan": "婚姻",
        "學歷_mode_or_nan": "學歷",
        "被保人投保年齡_min": "最小投保年齡",
        "被保人投保年齡_max": "最大投保年齡",
        "被保人投保年齡_mean": "平均投保年齡",
        "被保人目前年齡_last_valid": "被保人目前年齡",
        "要保人年收入_max": "要保人年收入",
        "生日_first": "生日",
    }

    customer_df = customer_df.rename(columns=rename_map)

    # =========================
    # 12. 衍生欄位
    # =========================
    customer_df["投保年資天數"] = (analysis_date - customer_df["首次投保日"]).dt.days
    customer_df["距離最近投保天數"] = (analysis_date - customer_df["最近投保日"]).dt.days

    # 比例欄位
    ratio_pairs = [
        ("含投資型保單數", "含投資型保單比例"),
        ("含躉繳保單數", "含躉繳保單比例"),
        ("含躉繳投資型保單數", "含躉繳投資型保單比例"),
        ("主約投資型保單數", "主約投資型保單比例"),
        ("主約躉繳保單數", "主約躉繳保單比例"),
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
    ]

    for num_col, ratio_col in ratio_pairs:
        if num_col in customer_df.columns:
            customer_df[ratio_col] = np.where(
                customer_df["保單數"] > 0,
                customer_df[num_col] / customer_df["保單數"],
                np.nan
            )

    # 是否曾經...
    ever_pairs = [
        ("含投資型保單數", "是否曾買過投資型"),
        ("含躉繳保單數", "是否曾買過躉繳"),
        ("含躉繳投資型保單數", "是否曾買過含躉繳投資型保單"),
        ("主約投資型保單數", "是否曾買過主約投資型"),
        ("主約躉繳保單數", "是否曾買過主約躉繳"),
        ("主約躉繳投資型保單數", "是否曾買過主約躉繳投資型"),
        ("躉繳投資型保單數", "是否曾買過躉繳投資型"),
    ]

    for num_col, flag_col in ever_pairs:
        if num_col in customer_df.columns:
            customer_df[flag_col] = (customer_df[num_col] > 0).astype("Int64")

    # 第一張是否就是躉繳投資型
    if "躉繳投資型保單數" in customer_df.columns:
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

    # 若有生日但沒有目前年齡，可補算
    if "被保人目前年齡" not in customer_df.columns and "生日" in customer_df.columns:
        customer_df["被保人目前年齡"] = (
            (analysis_date - customer_df["生日"]).dt.days / 365.25
        ).round(1)

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
        "性別" if "性別" in customer_df.columns else None,
        "婚姻" if "婚姻" in customer_df.columns else None,
        "學歷" if "學歷" in customer_df.columns else None,
        "生日" if "生日" in customer_df.columns else None,
        "被保人目前年齡" if "被保人目前年齡" in customer_df.columns else None,
        "要保人年收入" if "要保人年收入" in customer_df.columns else None,
        "要保人年收入級距" if "要保人年收入級距" in customer_df.columns else None,

        "目前營業單位" if "目前營業單位" in customer_df.columns else None,
        "目前經紀人1業代" if "目前經紀人1業代" in customer_df.columns else None,

        "保單數",
        "首次投保日",
        "最近投保日",
        "投保年資天數",
        "距離最近投保天數",

        "累計保單總保費",
        "平均每張保單保費",
        "保單保費中位數",
        "最大單張保單保費",
        "平均主約保費" if "平均主約保費" in customer_df.columns else None,
        "主約保費中位數" if "主約保費中位數" in customer_df.columns else None,
        "最大主約保費" if "最大主約保費" in customer_df.columns else None,

        "累計保單總繳款FYC",
        "平均每張保單繳款FYC",
        "最大單張保單繳款FYC",

        "含投資型保單數" if "含投資型保單數" in customer_df.columns else None,
        "含躉繳保單數" if "含躉繳保單數" in customer_df.columns else None,
        "含躉繳投資型保單數" if "含躉繳投資型保單數" in customer_df.columns else None,
        "主約投資型保單數" if "主約投資型保單數" in customer_df.columns else None,
        "主約躉繳保單數" if "主約躉繳保單數" in customer_df.columns else None,
        "主約躉繳投資型保單數" if "主約躉繳投資型保單數" in customer_df.columns else None,
        "躉繳投資型保單數" if "躉繳投資型保單數" in customer_df.columns else None,

        "是否曾買過投資型" if "是否曾買過投資型" in customer_df.columns else None,
        "是否曾買過躉繳" if "是否曾買過躉繳" in customer_df.columns else None,
        "是否曾買過躉繳投資型" if "是否曾買過躉繳投資型" in customer_df.columns else None,

        "首次躉繳投資型保單序號" if "首次躉繳投資型保單序號" in customer_df.columns else None,
        "首次躉繳投資型投保日" if "首次躉繳投資型投保日" in customer_df.columns else None,
        "第一張就買躉繳投資型" if "第一張就買躉繳投資型" in customer_df.columns else None,

        "近1年保單數",
        "近2年保單數",
        "近3年保單數",
        "高保費保單數_P75",
        "高保費保單數_P90",

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

print(customer_df.shape)
print(customer_df.head())
print(customer_df.columns.tolist())