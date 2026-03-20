# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 14:36:21 2026

@author: Z01788
"""

import pandas as pd
import numpy as np
import re


def _safe_numeric(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _safe_datetime(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    return df


def _sanitize_text_for_col(x):
    x = str(x)
    x = re.sub(r"[^\w\u4e00-\u9fff]+", "_", x)
    x = re.sub(r"_+", "_", x).strip("_")
    return x


def build_benchmark_rule_profile(
    benchmark_snapshot_df: pd.DataFrame,
    benchmark_policy_snapshot_df: pd.DataFrame,
    benchmark_first_df: pd.DataFrame,
    product_col: str = "主約商品險種主類別"
):
    """
    建立 benchmark-driven rule score 所需的 profile
    回傳：
    1. benchmark_profile: dict，可直接拿去算分
    2. benchmark_profile_tables: dict，各種摘要表，方便直接看/報告
    """

    bench_cust = benchmark_snapshot_df.copy()
    bench_pol = benchmark_policy_snapshot_df.copy()
    bench_first = benchmark_first_df.copy()

    for df in [bench_cust, bench_pol, bench_first]:
        df.columns = df.columns.str.strip()

    bench_cust = _safe_numeric(
        bench_cust,
        [
            "保單數",
            "距離首次買躉繳投資型前最近一次投保天數",
            "最大單張保單保費",
            "平均每張保單保費",
            "累計保單總保費"
        ]
    )

    bench_first = _safe_numeric(bench_first, ["首次躉繳投資型保單序號"])
    bench_pol = _safe_datetime(bench_pol, ["投保日"])

    # --------------------------------------------------
    # 1. 保單成熟度：首次躉繳投資型保單序號分布
    # --------------------------------------------------
    seq_dist_df = (
        bench_first["首次躉繳投資型保單序號"]
        .dropna()
        .astype(int)
        .value_counts()
        .sort_index()
        .reset_index()
    )
    seq_dist_df.columns = ["首次躉繳投資型保單序號", "客戶數"]
    seq_dist_df["比例"] = seq_dist_df["客戶數"] / seq_dist_df["客戶數"].sum()

    # 4張以上合併版本（用在 rule score 比較穩）
    seq_dist_bucket_df = seq_dist_df.copy()
    seq_dist_bucket_df["保單數分箱"] = seq_dist_bucket_df["首次躉繳投資型保單序號"].clip(upper=4)
    seq_dist_bucket_df = (
        seq_dist_bucket_df.groupby("保單數分箱", as_index=False)["客戶數"]
        .sum()
    )
    seq_dist_bucket_df["比例"] = seq_dist_bucket_df["客戶數"] / seq_dist_bucket_df["客戶數"].sum()

    seq_weight_map = dict(zip(seq_dist_bucket_df["保單數分箱"], seq_dist_bucket_df["比例"]))

    # --------------------------------------------------
    # 2. 購買前保單數分布（如果你想補充報告可用）
    # --------------------------------------------------
    pre_policy_dist_df = None
    if "保單數" in bench_cust.columns:
        pre_policy_dist_df = (
            bench_cust["保單數"]
            .dropna()
            .astype(int)
            .value_counts()
            .sort_index()
            .reset_index()
        )
        pre_policy_dist_df.columns = ["購買前保單數", "客戶數"]
        pre_policy_dist_df["比例"] = pre_policy_dist_df["客戶數"] / pre_policy_dist_df["客戶數"].sum()

    # --------------------------------------------------
    # 3. 商品主類別滲透率（購買前是否曾買過）
    # --------------------------------------------------
    required_pol_cols = ["被保人身分證字號", product_col]
    missing_pol_cols = [c for c in required_pol_cols if c not in bench_pol.columns]
    if missing_pol_cols:
        raise ValueError(f"benchmark_policy_snapshot_df 缺少欄位: {missing_pol_cols}")

    prod_presence = (
        bench_pol.assign(flag=1)
        .pivot_table(
            index="被保人身分證字號",
            columns=product_col,
            values="flag",
            aggfunc="max",
            fill_value=0
        )
    )

    product_rate_df = (
        prod_presence.mean()
        .sort_values(ascending=False)
        .reset_index()
    )
    product_rate_df.columns = [product_col, "客戶滲透率"]

    product_rate_map = dict(zip(product_rate_df[product_col], product_rate_df["客戶滲透率"]))

    # --------------------------------------------------
    # 4. 活躍度 / 購買時機 profile
    # --------------------------------------------------
    active_profile = {}
    active_summary_df = None

    if "距離首次買躉繳投資型前最近一次投保天數" in bench_cust.columns:
        s = bench_cust["距離首次買躉繳投資型前最近一次投保天數"].dropna()
        if len(s) > 0:
            active_profile = {
                "q25": s.quantile(0.25),
                "q50": s.quantile(0.50),
                "q75": s.quantile(0.75),
                "mean": s.mean(),
                "median": s.median()
            }
            active_summary_df = pd.DataFrame(
                {
                    "指標": ["q25", "q50", "q75", "mean", "median"],
                    "值": [
                        active_profile["q25"],
                        active_profile["q50"],
                        active_profile["q75"],
                        active_profile["mean"],
                        active_profile["median"]
                    ]
                }
            )

    # --------------------------------------------------
    # 5. 消費力 profile
    # --------------------------------------------------
    spending_cols = [
        "最大單張保單保費",
        "平均每張保單保費",
        "累計保單總保費"
    ]

    spending_profile_rows = []
    spending_profile_map = {}

    for c in spending_cols:
        if c in bench_cust.columns:
            s = bench_cust[c].dropna()
            if len(s) > 0:
                spending_profile_rows.append({
                    "欄位": c,
                    "平均": s.mean(),
                    "中位數": s.median(),
                    "P75": s.quantile(0.75),
                    "P90": s.quantile(0.90),
                    "最大值": s.max(),
                    "非空值數": s.notna().sum()
                })
                spending_profile_map[c] = {
                    "mean": s.mean(),
                    "median": s.median(),
                    "p75": s.quantile(0.75),
                    "p90": s.quantile(0.90),
                }

    spending_summary_df = pd.DataFrame(spending_profile_rows)

    # --------------------------------------------------
    # 組合 profile
    # --------------------------------------------------
    benchmark_profile = {
        "seq_weight_map": seq_weight_map,
        "seq_dist_df": seq_dist_df,
        "seq_dist_bucket_df": seq_dist_bucket_df,
        "product_rate_map": product_rate_map,
        "product_rate_df": product_rate_df,
        "active_profile": active_profile,
        "active_summary_df": active_summary_df,
        "spending_profile_map": spending_profile_map,
        "spending_summary_df": spending_summary_df,
        "product_col": product_col
    }

    benchmark_profile_tables = {
        "seq_dist_df": seq_dist_df,
        "seq_dist_bucket_df": seq_dist_bucket_df,
        "pre_policy_dist_df": pre_policy_dist_df,
        "product_rate_df": product_rate_df,
        "active_summary_df": active_summary_df,
        "spending_summary_df": spending_summary_df
    }

    return benchmark_profile, benchmark_profile_tables


benchmark_profile, benchmark_profile_tables = build_benchmark_rule_profile(
    benchmark_snapshot_df=benchmark_snapshot_df,
    benchmark_policy_snapshot_df=benchmark_policy_snapshot_df,
    benchmark_first_df=benchmark_first_df,
    product_col="主約商品險種主類別"
)


seq_dist_df = benchmark_profile_tables["seq_dist_df"]
seq_dist_bucket_df = benchmark_profile_tables["seq_dist_bucket_df"]
pre_policy_dist_df = benchmark_profile_tables["pre_policy_dist_df"]
product_rate_df = benchmark_profile_tables["product_rate_df"]
active_summary_df = benchmark_profile_tables["active_summary_df"]
spending_summary_df = benchmark_profile_tables["spending_summary_df"]



# %% (no-weight) 算 benchmark 的參考值

def _safe_numeric(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _safe_datetime(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    return df


def _sanitize_text_for_col(x):
    x = str(x)
    x = re.sub(r"[^\w\u4e00-\u9fff]+", "_", x)
    x = re.sub(r"_+", "_", x).strip("_")
    return x


def build_benchmark_rule_profile(
    benchmark_snapshot_df: pd.DataFrame,
    benchmark_policy_snapshot_df: pd.DataFrame,
    benchmark_first_df: pd.DataFrame,
    product_col: str = "主約商品險種主類別",
    top_n_routes: int = 30
):
    """
    建立 rule score 會用到的 benchmark profile
    回傳 7 張表：
    1. benchmark_summary_df
    2. seq_dist_df
    3. product_rate_df
    4. timing_profile_df
    5. spending_profile_df
    6. route_profile_df
    7. frequency_profile_df
    """

    bench_cust = benchmark_snapshot_df.copy()
    bench_pol = benchmark_policy_snapshot_df.copy()
    bench_first = benchmark_first_df.copy()

    for df in [bench_cust, bench_pol, bench_first]:
        df.columns = df.columns.str.strip()

    bench_cust = _safe_numeric(
        bench_cust,
        [
            "保單數",
            "距離首次買躉繳投資型前最近一次投保天數",
            "最大單張保單保費",
            "平均每張保單保費",
            "累計保單總保費"
        ]
    )

    bench_first = _safe_numeric(bench_first, ["首次躉繳投資型保單序號"])
    bench_pol = _safe_datetime(bench_pol, ["投保日"])
    bench_pol = _safe_numeric(bench_pol, ["保單序號"])

    # =========================
    # 1. benchmark 整體摘要
    # =========================
    summary_rows = []
    summary_targets = [
        "保單數",
        "距離首次買躉繳投資型前最近一次投保天數",
        "最大單張保單保費",
        "平均每張保單保費",
        "累計保單總保費",
    ]

    for c in summary_targets:
        if c in bench_cust.columns:
            summary_rows.append({
                "欄位": c,
                "樣本數": int(bench_cust[c].notna().sum()),
                "平均數": bench_cust[c].mean(),
                "中位數": bench_cust[c].median(),
                "P25": bench_cust[c].quantile(0.25),
                "P75": bench_cust[c].quantile(0.75),
                "最大值": bench_cust[c].max(),
            })

    benchmark_summary_df = pd.DataFrame(summary_rows)

    # =========================
    # 2. 第幾張才買（seq_dist）
    # =========================
    seq_dist_df = (
        bench_first["首次躉繳投資型保單序號"]
        .dropna()
        .astype(int)
        .value_counts()
        .sort_index()
        .reset_index()
    )
    seq_dist_df.columns = ["首次躉繳投資型保單序號", "客戶數"]
    seq_dist_df["比例"] = seq_dist_df["客戶數"] / seq_dist_df["客戶數"].sum()

    # =========================
    # 3. 商品主類別滲透率
    # =========================
    required_pol_cols = ["被保人身分證字號", product_col]
    missing_pol_cols = [c for c in required_pol_cols if c not in bench_pol.columns]
    if missing_pol_cols:
        raise ValueError(f"benchmark_policy_snapshot_df 缺少欄位: {missing_pol_cols}")

    product_presence = (
        bench_pol.assign(flag=1)
        .pivot_table(
            index="被保人身分證字號",
            columns=product_col,
            values="flag",
            aggfunc="max",
            fill_value=0
        )
    )

    product_rate_df = (
        product_presence.mean()
        .sort_values(ascending=False)
        .reset_index()
    )
    product_rate_df.columns = ["商品主類別", "滲透率"]

    # =========================
    # 4. timing profile
    # =========================
    timing_profile = {}

    if "距離首次買躉繳投資型前最近一次投保天數" in bench_cust.columns:
        s = bench_cust["距離首次買躉繳投資型前最近一次投保天數"].dropna()
        if len(s) > 0:
            timing_profile["最近一次投保距購買日_平均"] = s.mean()
            timing_profile["最近一次投保距購買日_中位數"] = s.median()
            timing_profile["最近一次投保距購買日_P25"] = s.quantile(0.25)
            timing_profile["最近一次投保距購買日_P75"] = s.quantile(0.75)

    if {"被保人身分證字號", "投保日", "保單序號"}.issubset(bench_pol.columns):
        tmp = bench_pol.sort_values(["被保人身分證字號", "投保日", "保單序號"]).copy()
        tmp["前一張投保日"] = tmp.groupby("被保人身分證字號")["投保日"].shift(1)
        tmp["與前一張保單間距天數"] = (tmp["投保日"] - tmp["前一張投保日"]).dt.days
        gap_s = tmp["與前一張保單間距天數"].dropna()
        if len(gap_s) > 0:
            timing_profile["買前保單間距_平均"] = gap_s.mean()
            timing_profile["買前保單間距_中位數"] = gap_s.median()
            timing_profile["買前保單間距_P25"] = gap_s.quantile(0.25)
            timing_profile["買前保單間距_P75"] = gap_s.quantile(0.75)

    timing_profile_df = pd.DataFrame(
        [{"指標": k, "值": v} for k, v in timing_profile.items()]
    )
    
    # =========================
    # 4-2. 最後一張距離天數 × 保單數（加權）
    # =========================
    timing_by_seq_rows = []
    
    if {
        "被保人身分證字號",
        "距離首次買躉繳投資型前最近一次投保天數"
    }.issubset(bench_cust.columns):
    
        # 如果 bench_cust 本來就有同名欄位，先刪掉再 merge，避免 _x / _y
        bench_cust_tmp = bench_cust.drop(columns=["首次躉繳投資型保單序號"], errors="ignore").copy()
    
        if "首次躉繳投資型保單序號" not in bench_first.columns:
            print("warning: benchmark_first_df 缺少『首次躉繳投資型保單序號』，timing_by_seq_df 將為空表")
            timing_by_seq_df = pd.DataFrame()
        else:
            seq_timing_df = bench_cust_tmp.merge(
                bench_first[["被保人身分證字號", "首次躉繳投資型保單序號"]],
                on="被保人身分證字號",
                how="left"
            )
    
            seq_timing_df["首次躉繳投資型保單序號"] = pd.to_numeric(
                seq_timing_df["首次躉繳投資型保單序號"], errors="coerce"
            )
    
            for seq_no in sorted(seq_timing_df["首次躉繳投資型保單序號"].dropna().unique()):
                sub = seq_timing_df.loc[
                    seq_timing_df["首次躉繳投資型保單序號"] == seq_no,
                    "距離首次買躉繳投資型前最近一次投保天數"
                ].dropna()
    
                if len(sub) > 0:
                    timing_by_seq_rows.append({
                        "首次躉繳投資型保單序號": int(seq_no),
                        "樣本數": len(sub),
                        "平均最後一張距離天數": sub.mean(),
                        "中位數最後一張距離天數": sub.median(),
                        "P25": sub.quantile(0.25),
                        "P75": sub.quantile(0.75),
                    })
    
            timing_by_seq_df = pd.DataFrame(timing_by_seq_rows)
    else:
        timing_by_seq_df = pd.DataFrame()

    # =========================
    # 5. spending profile
    # =========================
    spending_rows = []
    spending_targets = [
        "最大單張保單保費",
        "平均每張保單保費",
        "累計保單總保費",
    ]

    for c in spending_targets:
        if c in bench_cust.columns:
            s = bench_cust[c].dropna()
            if len(s) > 0:
                spending_rows.append({
                    "欄位": c,
                    "平均數": s.mean(),
                    "中位數": s.median(),
                    "P25": s.quantile(0.25),
                    "P75": s.quantile(0.75),
                    "P90": s.quantile(0.90),
                })

    spending_profile_df = pd.DataFrame(spending_rows)

    # =========================
    # 6. route profile
    # =========================
    route_profile_parts = []

    if {"被保人身分證字號", "投保日", "保單序號", product_col}.issubset(bench_pol.columns):
        tmp = bench_pol.sort_values(["被保人身分證字號", "投保日", "保單序號"]).copy()

        # 前一張商品
        tmp["前一張商品主類別"] = tmp.groupby("被保人身分證字號")[product_col].shift(1)
        prev_major_df = (
            tmp["前一張商品主類別"]
            .value_counts(dropna=False)
            .reset_index()
        )
        prev_major_df.columns = ["商品主類別", "客戶數"]
        prev_major_df["比例"] = prev_major_df["客戶數"] / prev_major_df["客戶數"].sum()
        prev_major_df["profile_type"] = "前一張商品主類別"

        route_profile_parts.append(prev_major_df)

        # 前三張路徑
        tmp["買前總保單數"] = tmp.groupby("被保人身分證字號")["被保人身分證字號"].transform("count")
        tmp["買前順序"] = tmp.groupby("被保人身分證字號").cumcount() + 1
        tmp["距離最後一張排序"] = tmp["買前總保單數"] - tmp["買前順序"] + 1

        last3 = tmp[tmp["距離最後一張排序"] <= 3].copy()
        last3["前三張內順序"] = last3.groupby("被保人身分證字號").cumcount() + 1

        route_detail_df = (
            last3.pivot_table(
                index="被保人身分證字號",
                columns="前三張內順序",
                values=product_col,
                aggfunc="first"
            )
            .reset_index()
        )

        route_detail_df = route_detail_df.rename(columns={
            1: "前第1張商品",
            2: "前第2張商品",
            3: "前第3張商品",
        })

        for c in ["前第1張商品", "前第2張商品", "前第3張商品"]:
            if c not in route_detail_df.columns:
                route_detail_df[c] = np.nan
            route_detail_df[c] = route_detail_df[c].fillna("Start")

        route_detail_df["前三張保單路徑"] = (
            route_detail_df["前第1張商品"] + " → " +
            route_detail_df["前第2張商品"] + " → " +
            route_detail_df["前第3張商品"] + " → 躉繳投資型"
        )

        top_route_df = (
            route_detail_df["前三張保單路徑"]
            .value_counts()
            .reset_index()
        )
        top_route_df.columns = ["路徑", "客戶數"]
        top_route_df["比例"] = top_route_df["客戶數"] / top_route_df["客戶數"].sum()
        top_route_df["profile_type"] = "前三張保單路徑"
        top_route_df = top_route_df.head(top_n_routes)

        route_profile_parts.append(top_route_df)

    route_profile_df = pd.concat(route_profile_parts, ignore_index=True) if route_profile_parts else pd.DataFrame()

    # =========================
    # 7. frequency profile
    # =========================
    frequency_rows = []

    if {"被保人身分證字號", "投保日", "保單序號"}.issubset(bench_pol.columns):
        tmp = bench_pol.sort_values(["被保人身分證字號", "投保日", "保單序號"]).copy()
        tmp["前一張投保日"] = tmp.groupby("被保人身分證字號")["投保日"].shift(1)
        tmp["與前一張保單間距天數"] = (tmp["投保日"] - tmp["前一張投保日"]).dt.days
        tmp["買前總保單數"] = tmp.groupby("被保人身分證字號")["被保人身分證字號"].transform("count")
        tmp["買前順序"] = tmp.groupby("被保人身分證字號").cumcount() + 1
        tmp["距離躉繳投資型前的位置"] = tmp["買前總保單數"] - tmp["買前順序"] + 1

        for pos in sorted(tmp["距離躉繳投資型前的位置"].dropna().unique()):
            s = tmp.loc[tmp["距離躉繳投資型前的位置"] == pos, "與前一張保單間距天數"].dropna()
            if len(s) > 0:
                frequency_rows.append({
                    "距離躉繳投資型前的位置": int(pos),
                    "平均間距天數": s.mean(),
                    "中位數間距天數": s.median(),
                    "P25": s.quantile(0.25),
                    "P75": s.quantile(0.75),
                    "樣本數": len(s)
                })

    frequency_profile_df = pd.DataFrame(frequency_rows)

    return (
        benchmark_summary_df,
        seq_dist_df,
        product_rate_df,
        timing_profile_df,
        timing_by_seq_df, 
        spending_profile_df,
        route_profile_df,
        frequency_profile_df
    )


# %% 算 benchmark 的參考值

def _safe_numeric(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _safe_datetime(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    return df


def _sanitize_text_for_col(x):
    x = str(x)
    x = re.sub(r"[^\w\u4e00-\u9fff]+", "_", x)
    x = re.sub(r"_+", "_", x).strip("_")
    return x


def weighted_quantile(values, quantiles, sample_weight=None):
    """
    加權分位數
    quantiles 可傳單一值或 list-like，範圍 0~1
    """
    values = np.asarray(values, dtype=float)
    quantiles = np.atleast_1d(quantiles)

    if sample_weight is None:
        sample_weight = np.ones(len(values))
    sample_weight = np.asarray(sample_weight, dtype=float)

    mask = ~np.isnan(values) & ~np.isnan(sample_weight)
    values = values[mask]
    sample_weight = sample_weight[mask]

    if len(values) == 0:
        if len(quantiles) == 1:
            return np.nan
        return np.array([np.nan] * len(quantiles))

    sorter = np.argsort(values)
    values = values[sorter]
    sample_weight = sample_weight[sorter]

    weighted_cdf = np.cumsum(sample_weight) - 0.5 * sample_weight
    weighted_cdf = weighted_cdf / np.sum(sample_weight)

    result = np.interp(quantiles, weighted_cdf, values)

    if len(result) == 1:
        return result[0]
    return result


def build_benchmark_rule_profile(
    benchmark_snapshot_df: pd.DataFrame,
    benchmark_policy_snapshot_df: pd.DataFrame,
    benchmark_first_df: pd.DataFrame,
    product_col: str = "主約商品險種主類別",
    top_n_routes: int = 30,
    decay_lambda: float = 0.35,
    reference_date=None
):
    """
    建立時間加權版 benchmark rule profile（年衰減權重）

    參數
    ----------
    benchmark_snapshot_df : DataFrame
        benchmark 客戶在買前的 customer-level snapshot

    benchmark_policy_snapshot_df : DataFrame
        benchmark 客戶在買前的 policy-level snapshot

    benchmark_first_df : DataFrame
        每位 benchmark 客戶首次買躉繳投資型的資訊
        至少需包含：
        - 被保人身分證字號
        - 首次躉繳投資型投保日
        - 首次躉繳投資型保單序號

    product_col : str
        商品主類別欄位名稱

    top_n_routes : int
        前幾名熱門路徑

    decay_lambda : float
        年衰減係數，越大代表舊資料衰減越快

    reference_date : str / Timestamp / None
        權重基準日
        若 None，預設用 benchmark_first_df 的最大首次購買日

    回傳
    ----------
    benchmark_summary_df
    seq_dist_df
    product_rate_df
    timing_profile_df
    spending_profile_df
    route_profile_df
    frequency_profile_df
    """

    bench_cust = benchmark_snapshot_df.copy()
    bench_pol = benchmark_policy_snapshot_df.copy()
    bench_first = benchmark_first_df.copy()

    for df in [bench_cust, bench_pol, bench_first]:
        df.columns = df.columns.str.strip()

    # =========================
    # 0. 基本檢查
    # =========================
    required_first_cols = ["被保人身分證字號", "首次躉繳投資型投保日", "首次躉繳投資型保單序號"]
    missing_first = [c for c in required_first_cols if c not in bench_first.columns]
    if missing_first:
        raise ValueError(f"benchmark_first_df 缺少必要欄位: {missing_first}")

    required_pol_cols = ["被保人身分證字號", product_col]
    missing_pol = [c for c in required_pol_cols if c not in bench_pol.columns]
    if missing_pol:
        raise ValueError(f"benchmark_policy_snapshot_df 缺少必要欄位: {missing_pol}")

    # =========================
    # 1. 型別整理
    # =========================
    bench_cust = _safe_numeric(
        bench_cust,
        [
            "保單數",
            "距離首次買躉繳投資型前最近一次投保天數",
            "最大單張保單保費",
            "平均每張保單保費",
            "累計保單總保費"
        ]
    )

    bench_first = _safe_numeric(bench_first, ["首次躉繳投資型保單序號"])
    bench_first = _safe_datetime(bench_first, ["首次躉繳投資型投保日"])

    bench_pol = _safe_datetime(bench_pol, ["投保日"])
    bench_pol = _safe_numeric(bench_pol, ["保單序號"])

    # =========================
    # 2. 建立時間權重（年衰減）
    # =========================
    if reference_date is None:
        reference_date = bench_first["首次躉繳投資型投保日"].max()
    else:
        reference_date = pd.to_datetime(reference_date)

    bench_first["距離基準日年數"] = (
        (reference_date - bench_first["首次躉繳投資型投保日"]).dt.days / 365.25
    )

    # 年衰減權重：越新越接近1，越舊越接近0
    bench_first["時間權重"] = np.exp(-decay_lambda * bench_first["距離基準日年數"])

    # merge 回 snapshot / policy
    bench_cust = bench_cust.drop(columns=["時間權重"], errors="ignore")
    bench_cust = bench_cust.merge(
        bench_first[["被保人身分證字號", "時間權重"]].drop_duplicates("被保人身分證字號"),
        on="被保人身分證字號",
        how="left"
    )

    bench_pol = bench_pol.drop(columns=["時間權重"], errors="ignore")
    bench_pol = bench_pol.merge(
        bench_first[["被保人身分證字號", "時間權重"]].drop_duplicates("被保人身分證字號"),
        on="被保人身分證字號",
        how="left"
    )

    # =========================
    # 3. benchmark 整體摘要（加權）
    # =========================
    summary_rows = []
    summary_targets = [
        "保單數",
        "距離首次買躉繳投資型前最近一次投保天數",
        "最大單張保單保費",
        "平均每張保單保費",
        "累計保單總保費",
    ]

    for c in summary_targets:
        if c in bench_cust.columns:
            tmp = bench_cust[[c, "時間權重"]].dropna()
            if len(tmp) > 0:
                values = tmp[c].values
                weights = tmp["時間權重"].values
                summary_rows.append({
                    "欄位": c,
                    "樣本數": len(tmp),
                    "加權平均數": np.average(values, weights=weights),
                    "加權中位數": weighted_quantile(values, 0.5, weights),
                    "加權P25": weighted_quantile(values, 0.25, weights),
                    "加權P75": weighted_quantile(values, 0.75, weights),
                    "最大值": np.nanmax(values),
                })

    benchmark_summary_df = pd.DataFrame(summary_rows)

    # =========================
    # 4. 第幾張才買（seq_dist，加權）
    # =========================
    seq_dist_df = (
        bench_first.dropna(subset=["首次躉繳投資型保單序號", "時間權重"])
        .assign(首次躉繳投資型保單序號=lambda x: x["首次躉繳投資型保單序號"].astype(int))
        .groupby("首次躉繳投資型保單序號", as_index=False)["時間權重"]
        .sum()
    )
    seq_dist_df = seq_dist_df.rename(columns={"時間權重": "加權客戶數"})
    seq_dist_df["比例"] = seq_dist_df["加權客戶數"] / seq_dist_df["加權客戶數"].sum()

    # =========================
    # 5. 商品主類別滲透率（加權）
    # =========================
    product_presence = (
        bench_pol.assign(flag=1)
        .pivot_table(
            index="被保人身分證字號",
            columns=product_col,
            values="flag",
            aggfunc="max",
            fill_value=0
        )
        .reset_index()
    )

    product_presence = product_presence.merge(
        bench_first[["被保人身分證字號", "時間權重"]].drop_duplicates("被保人身分證字號"),
        on="被保人身分證字號",
        how="left"
    )

    prod_cols = [c for c in product_presence.columns if c not in ["被保人身分證字號", "時間權重"]]

    product_rate_rows = []
    for c in prod_cols:
        tmp = product_presence[[c, "時間權重"]].dropna()
        if len(tmp) > 0:
            weighted_rate = np.average(tmp[c].values, weights=tmp["時間權重"].values)
            product_rate_rows.append({
                "商品主類別": c,
                "滲透率": weighted_rate
            })

    product_rate_df = pd.DataFrame(product_rate_rows).sort_values("滲透率", ascending=False)

    # =========================
    # 6. timing profile（加權）
    # =========================
    timing_profile = {}

    # 最近一次投保距購買日
    if "距離首次買躉繳投資型前最近一次投保天數" in bench_cust.columns:
        tmp = bench_cust[["距離首次買躉繳投資型前最近一次投保天數", "時間權重"]].dropna()
        if len(tmp) > 0:
            values = tmp["距離首次買躉繳投資型前最近一次投保天數"].values
            weights = tmp["時間權重"].values

            timing_profile["最近一次投保距購買日_平均"] = np.average(values, weights=weights)
            timing_profile["最近一次投保距購買日_中位數"] = weighted_quantile(values, 0.5, weights)
            timing_profile["最近一次投保距購買日_P25"] = weighted_quantile(values, 0.25, weights)
            timing_profile["最近一次投保距購買日_P75"] = weighted_quantile(values, 0.75, weights)

    # 買前保單間距
    if {"被保人身分證字號", "投保日", "保單序號"}.issubset(bench_pol.columns):
        tmp = bench_pol.sort_values(["被保人身分證字號", "投保日", "保單序號"]).copy()
        tmp["前一張投保日"] = tmp.groupby("被保人身分證字號")["投保日"].shift(1)
        tmp["與前一張保單間距天數"] = (tmp["投保日"] - tmp["前一張投保日"]).dt.days

        tmp2 = tmp[["與前一張保單間距天數", "時間權重"]].dropna()
        if len(tmp2) > 0:
            values = tmp2["與前一張保單間距天數"].values
            weights = tmp2["時間權重"].values

            timing_profile["買前保單間距_平均"] = np.average(values, weights=weights)
            timing_profile["買前保單間距_中位數"] = weighted_quantile(values, 0.5, weights)
            timing_profile["買前保單間距_P25"] = weighted_quantile(values, 0.25, weights)
            timing_profile["買前保單間距_P75"] = weighted_quantile(values, 0.75, weights)

    timing_profile_df = pd.DataFrame(
        [{"指標": k, "值": v} for k, v in timing_profile.items()]
    )

    # =========================
    # 7. spending profile（加權）
    # =========================
    spending_rows = []
    spending_targets = [
        "最大單張保單保費",
        "平均每張保單保費",
        "累計保單總保費",
    ]

    for c in spending_targets:
        if c in bench_cust.columns:
            tmp = bench_cust[[c, "時間權重"]].dropna()
            if len(tmp) > 0:
                values = tmp[c].values
                weights = tmp["時間權重"].values

                spending_rows.append({
                    "欄位": c,
                    "加權平均數": np.average(values, weights=weights),
                    "加權中位數": weighted_quantile(values, 0.5, weights),
                    "加權P25": weighted_quantile(values, 0.25, weights),
                    "加權P75": weighted_quantile(values, 0.75, weights),
                    "加權P90": weighted_quantile(values, 0.90, weights),
                })

    spending_profile_df = pd.DataFrame(spending_rows)

    # =========================
    # 8. route profile（加權）
    # =========================
    route_profile_parts = []

    if {"被保人身分證字號", "投保日", "保單序號", product_col}.issubset(bench_pol.columns):
        tmp = bench_pol.sort_values(["被保人身分證字號", "投保日", "保單序號"]).copy()

        # 前一張商品主類別（取每位客戶最後一張買前保單的前一張）
        tmp["前一張商品主類別"] = tmp.groupby("被保人身分證字號")[product_col].shift(1)

        prev_df = (
            tmp.groupby("被保人身分證字號", as_index=False)
            .tail(1)[["被保人身分證字號", "前一張商品主類別", "時間權重"]]
        )

        prev_major_df = (
            prev_df.groupby("前一張商品主類別", dropna=False, as_index=False)["時間權重"]
            .sum()
            .rename(columns={
                "前一張商品主類別": "商品主類別",
                "時間權重": "加權客戶數"
            })
        )
        prev_major_df["比例"] = prev_major_df["加權客戶數"] / prev_major_df["加權客戶數"].sum()
        prev_major_df["profile_type"] = "前一張商品主類別"

        route_profile_parts.append(prev_major_df)

        # 前三張路徑
        tmp["買前總保單數"] = tmp.groupby("被保人身分證字號")["被保人身分證字號"].transform("count")
        tmp["買前順序"] = tmp.groupby("被保人身分證字號").cumcount() + 1
        tmp["距離最後一張排序"] = tmp["買前總保單數"] - tmp["買前順序"] + 1

        last3 = tmp[tmp["距離最後一張排序"] <= 3].copy()
        last3["前三張內順序"] = last3.groupby("被保人身分證字號").cumcount() + 1

        route_detail_df = (
            last3.pivot_table(
                index="被保人身分證字號",
                columns="前三張內順序",
                values=product_col,
                aggfunc="first"
            )
            .reset_index()
        )

        route_detail_df = route_detail_df.rename(columns={
            1: "前第1張商品",
            2: "前第2張商品",
            3: "前第3張商品",
        })

        for c in ["前第1張商品", "前第2張商品", "前第3張商品"]:
            if c not in route_detail_df.columns:
                route_detail_df[c] = np.nan
            route_detail_df[c] = route_detail_df[c].fillna("Start")

        route_detail_df["前三張保單路徑"] = (
            route_detail_df["前第1張商品"] + " → " +
            route_detail_df["前第2張商品"] + " → " +
            route_detail_df["前第3張商品"] + " → 躉繳投資型"
        )

        route_detail_df = route_detail_df.merge(
            bench_first[["被保人身分證字號", "時間權重"]].drop_duplicates("被保人身分證字號"),
            on="被保人身分證字號",
            how="left"
        )

        top_route_df = (
            route_detail_df.groupby("前三張保單路徑", as_index=False)["時間權重"]
            .sum()
            .rename(columns={
                "前三張保單路徑": "路徑",
                "時間權重": "加權客戶數"
            })
        )
        top_route_df["比例"] = top_route_df["加權客戶數"] / top_route_df["加權客戶數"].sum()
        top_route_df["profile_type"] = "前三張保單路徑"
        top_route_df = top_route_df.sort_values("比例", ascending=False).head(top_n_routes)

        route_profile_parts.append(top_route_df)

    route_profile_df = pd.concat(route_profile_parts, ignore_index=True) if route_profile_parts else pd.DataFrame()

    # =========================
    # 9. frequency profile（加權）
    # =========================
    frequency_rows = []

    if {"被保人身分證字號", "投保日", "保單序號"}.issubset(bench_pol.columns):
        tmp = bench_pol.sort_values(["被保人身分證字號", "投保日", "保單序號"]).copy()
        tmp["前一張投保日"] = tmp.groupby("被保人身分證字號")["投保日"].shift(1)
        tmp["與前一張保單間距天數"] = (tmp["投保日"] - tmp["前一張投保日"]).dt.days
        tmp["買前總保單數"] = tmp.groupby("被保人身分證字號")["被保人身分證字號"].transform("count")
        tmp["買前順序"] = tmp.groupby("被保人身分證字號").cumcount() + 1
        tmp["距離躉繳投資型前的位置"] = tmp["買前總保單數"] - tmp["買前順序"] + 1

        for pos in sorted(tmp["距離躉繳投資型前的位置"].dropna().unique()):
            sub = tmp.loc[
                tmp["距離躉繳投資型前的位置"] == pos,
                ["與前一張保單間距天數", "時間權重"]
            ].dropna()

            if len(sub) > 0:
                values = sub["與前一張保單間距天數"].values
                weights = sub["時間權重"].values

                frequency_rows.append({
                    "距離躉繳投資型前的位置": int(pos),
                    "加權平均間距天數": np.average(values, weights=weights),
                    "加權中位數間距天數": weighted_quantile(values, 0.5, weights),
                    "加權P25": weighted_quantile(values, 0.25, weights),
                    "加權P75": weighted_quantile(values, 0.75, weights),
                    "樣本數": len(sub)
                })

    frequency_profile_df = pd.DataFrame(frequency_rows)

    return (
        benchmark_summary_df,
        seq_dist_df,
        product_rate_df,
        timing_profile_df,
        spending_profile_df,
        route_profile_df,
        frequency_profile_df
    )


# %% apply_benchmark_profile_to_candidate()

def build_candidate_product_flags(
    policy_df: pd.DataFrame,
    candidate_df: pd.DataFrame,
    product_col: str = "主約商品險種主類別"
):
    pol = policy_df.copy()
    cand = candidate_df.copy()

    pol.columns = pol.columns.str.strip()
    cand.columns = cand.columns.str.strip()

    required_cols = ["被保人身分證字號", product_col]
    missing = [c for c in required_cols if c not in pol.columns]
    if missing:
        raise ValueError(f"policy_df 缺少欄位: {missing}")

    pol = pol[pol["被保人身分證字號"].isin(cand["被保人身分證字號"])].copy()

    prod_presence = (
        pol.assign(flag=1)
        .pivot_table(
            index="被保人身分證字號",
            columns=product_col,
            values="flag",
            aggfunc="max",
            fill_value=0
        )
        .reset_index()
    )

    rename_map = {}
    for c in prod_presence.columns:
        if c == "被保人身分證字號":
            continue
        rename_map[c] = f"是否曾買過_{_sanitize_text_for_col(c)}"

    prod_presence = prod_presence.rename(columns=rename_map)

    cand = cand.merge(prod_presence, on="被保人身分證字號", how="left")

    for c in cand.columns:
        if c.startswith("是否曾買過_"):
            cand[c] = cand[c].fillna(0)

    return cand, rename_map


def build_candidate_recent_route_frequency(policy_df: pd.DataFrame, candidate_df: pd.DataFrame, product_col: str = "主約商品險種主類別"):
    """
    幫 candidate 建：
    1. 前一張商品主類別
    2. 最近三張路徑
    3. 最近一張 / 二張的購買間距
    """
    pol = policy_df.copy()
    cand = candidate_df.copy()

    pol.columns = pol.columns.str.strip()
    cand.columns = cand.columns.str.strip()

    pol = _safe_datetime(pol, ["投保日"])
    pol = _safe_numeric(pol, ["保單序號"])

    pol = pol[pol["被保人身分證字號"].isin(cand["被保人身分證字號"])].copy()
    pol = pol.sort_values(["被保人身分證字號", "投保日", "保單序號"])

    # 前一張商品
    pol["前一張商品主類別"] = pol.groupby("被保人身分證字號")[product_col].shift(1)

    # 間距
    pol["前一張投保日"] = pol.groupby("被保人身分證字號")["投保日"].shift(1)
    pol["與前一張保單間距天數"] = (pol["投保日"] - pol["前一張投保日"]).dt.days

    # 最近三張
    pol["保單總數"] = pol.groupby("被保人身分證字號")["被保人身分證字號"].transform("count")
    pol["順序"] = pol.groupby("被保人身分證字號").cumcount() + 1
    pol["距離最近保單位置"] = pol["保單總數"] - pol["順序"] + 1

    recent3 = pol[pol["距離最近保單位置"] <= 3].copy()
    recent3["最近三張順序"] = recent3.groupby("被保人身分證字號").cumcount() + 1

    route_df = (
        recent3.pivot_table(
            index="被保人身分證字號",
            columns="最近三張順序",
            values=product_col,
            aggfunc="first"
        )
        .reset_index()
    )

    route_df = route_df.rename(columns={
        1: "最近第1張商品",
        2: "最近第2張商品",
        3: "最近第3張商品",
    })

    for c in ["最近第1張商品", "最近第2張商品", "最近第3張商品"]:
        if c not in route_df.columns:
            route_df[c] = "Start"
        route_df[c] = route_df[c].fillna("Start")

    route_df["最近三張路徑"] = (
        route_df["最近第1張商品"] + " → " +
        route_df["最近第2張商品"] + " → " +
        route_df["最近第3張商品"]
    )

    # 前一張商品 / 最新間距
    latest_df = (
        pol.groupby("被保人身分證字號", as_index=False)
        .tail(1)[["被保人身分證字號", "前一張商品主類別", "與前一張保單間距天數"]]
        .copy()
    )

    cand = cand.merge(route_df, on="被保人身分證字號", how="left")
    cand = cand.merge(latest_df, on="被保人身分證字號", how="left")

    return cand


def apply_benchmark_profile_to_candidate(
    candidate_df: pd.DataFrame,
    policy_df: pd.DataFrame,
    benchmark_summary_df: pd.DataFrame,
    seq_dist_df: pd.DataFrame,
    product_rate_df: pd.DataFrame,
    timing_profile_df: pd.DataFrame,
    spending_profile_df: pd.DataFrame,
    route_profile_df: pd.DataFrame,
    frequency_profile_df: pd.DataFrame,
    product_col: str = "主約商品險種主類別"
):
    df = candidate_df.copy()
    df.columns = df.columns.str.strip()

    # 數值整理
    for c in ["保單數", "距離最近投保天數", "最大單張保單保費", "平均每張保單保費", "累計保單總保費"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # 1. 商品旗標
    df, _ = build_candidate_product_flags(
        policy_df=policy_df,
        candidate_df=df,
        product_col=product_col
    )

    # 2. candidate 路徑 / 頻率欄位
    df = build_candidate_recent_route_frequency(
        policy_df=policy_df,
        candidate_df=df,
        product_col=product_col
    )

    # 3. 保單成熟度
    seq_weight = {}
    for _, row in seq_dist_df.iterrows():
        seq = int(row["首次躉繳投資型保單序號"])
        ratio = row["比例"]
        if seq >= 4:
            seq = 4
        seq_weight[seq] = seq_weight.get(seq, 0) + ratio

    df["保單數分箱"] = df["保單數"].clip(upper=4)
    df["benchmark_保單成熟度權重"] = df["保單數分箱"].map(seq_weight)

    # 4. 商品主類別滲透率
    product_rate_map = dict(zip(product_rate_df["商品主類別"], product_rate_df["滲透率"]))
    df["benchmark_商品路徑分數原始"] = 0.0
    df["benchmark_商品路徑權重總和"] = 0.0

    for raw_prod, rate in product_rate_map.items():
        flag_col = f"是否曾買過_{_sanitize_text_for_col(raw_prod)}"
        if flag_col in df.columns:
            df[flag_col] = pd.to_numeric(df[flag_col], errors="coerce").fillna(0)
            df["benchmark_商品路徑分數原始"] += df[flag_col] * rate
            df["benchmark_商品路徑權重總和"] += rate

    df["benchmark_商品路徑分數"] = np.where(
        df["benchmark_商品路徑權重總和"] > 0,
        df["benchmark_商品路徑分數原始"] / df["benchmark_商品路徑權重總和"],
        np.nan
    )

    # 5. timing 參考值
    timing_map = dict(zip(timing_profile_df["指標"], timing_profile_df["值"]))
    for k, v in timing_map.items():
        df[f"benchmark_{k}"] = v

    # 6. spending 參考值
    if not spending_profile_df.empty:
        for _, row in spending_profile_df.iterrows():
            base = row["欄位"]
            for metric in ["平均數", "中位數", "P25", "P75", "P90"]:
                df[f"benchmark_{base}_{metric}"] = row[metric]
    
    

    # 7. route profile：前一張商品主類別
    prev_route_df = route_profile_df[route_profile_df["profile_type"] == "前一張商品主類別"].copy()
    if not prev_route_df.empty and "前一張商品主類別" in df.columns:
        prev_map = dict(zip(prev_route_df["商品主類別"], prev_route_df["比例"]))
        df["benchmark_前一張商品主類別分數"] = df["前一張商品主類別"].map(prev_map)

    # 8. route profile：前三張路徑
    top_route_df = route_profile_df[route_profile_df["profile_type"] == "前三張保單路徑"].copy()
    if not top_route_df.empty and "最近三張路徑" in df.columns:
        # benchmark 路徑是「A → B → C → 躉繳投資型」，candidate 只留前段
        route_map = {}
        for _, row in top_route_df.iterrows():
            full_route = row["路徑"]
            ratio = row["比例"]
            short_route = full_route.replace(" → 躉繳投資型", "")
            route_map[short_route] = ratio

        df["benchmark_前三張路徑分數"] = df["最近三張路徑"].map(route_map)

    # 9. frequency profile：最近間距
    if not frequency_profile_df.empty and "與前一張保單間距天數" in df.columns:
        # 先用位置=1（最接近購買前的一張）當第一版
        freq1 = frequency_profile_df.loc[
            frequency_profile_df["距離躉繳投資型前的位置"] == 1
        ]
        if len(freq1) > 0:
            df["benchmark_最近間距_中位數"] = freq1["中位數間距天數"].iloc[0]
            df["benchmark_最近間距_P25"] = freq1["P25"].iloc[0]
            df["benchmark_最近間距_P75"] = freq1["P75"].iloc[0]

    return df


# %% 用參考值幫 candidate 算分

def build_rule_score_from_profile(
    candidate_rule_base_df: pd.DataFrame,
    weight_config: dict = None,
    bonus_weight_config: dict = None,
    use_sqrt_scaling: bool = True
):
    df = candidate_rule_base_df.copy()
    df.columns = df.columns.str.strip()

    if weight_config is None:
        weight_config = {
            "rule_保單成熟度": 0.30,
            "rule_商品路徑": 0.25,
            "rule_購買時機": 0.20,
            "rule_消費力": 0.15,
        }

    if bonus_weight_config is None:
        bonus_weight_config = {
            "rule_前一張商品相似度": 0.05,
            "rule_前三張路徑相似度": 0.03,
            "rule_購買頻率相似度": 0.02,
        }

    numeric_cols = [
        "benchmark_保單成熟度權重",
        "benchmark_商品路徑分數",
        "距離最近投保天數",
        "最大單張保單保費",
        "平均每張保單保費",
        "累計保單總保費",
        "benchmark_最近一次投保距購買日_P25",
        "benchmark_最近一次投保距購買日_中位數",
        "benchmark_最近一次投保距購買日_P75",
        "benchmark_最大單張保單保費_中位數",
        "benchmark_最大單張保單保費_P75",
        "benchmark_平均每張保單保費_中位數",
        "benchmark_平均每張保單保費_P75",
        "benchmark_累計保單總保費_中位數",
        "benchmark_累計保單總保費_P75",
        "benchmark_前一張商品主類別分數",
        "benchmark_前三張路徑分數",
        "benchmark_最近間距_中位數",
        "benchmark_最近間距_P25",
        "benchmark_最近間距_P75",
        "與前一張保單間距天數",
    ]

    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # 1. rule_保單成熟度
    if "benchmark_保單成熟度權重" in df.columns:
        s = df["benchmark_保單成熟度權重"].copy()
        if use_sqrt_scaling:
            s = np.sqrt(s)
        if s.notna().sum() > 0 and s.max() > s.min():
            df["rule_保單成熟度"] = (s - s.min()) / (s.max() - s.min())
        else:
            df["rule_保單成熟度"] = 1.0
    else:
        df["rule_保單成熟度"] = np.nan

    # 2. rule_商品路徑
    if "benchmark_商品路徑分數" in df.columns:
        s = df["benchmark_商品路徑分數"].copy()
        if use_sqrt_scaling:
            s = np.sqrt(s)
        if s.notna().sum() > 0 and s.max() > s.min():
            df["rule_商品路徑"] = (s - s.min()) / (s.max() - s.min())
        else:
            df["rule_商品路徑"] = 1.0
    else:
        df["rule_商品路徑"] = np.nan

    # 3. rule_購買時機
    def timing_score(row):
        x = row.get("距離最近投保天數", np.nan)
        q25 = row.get("benchmark_最近一次投保距購買日_P25", np.nan)
        med = row.get("benchmark_最近一次投保距購買日_中位數", np.nan)
        q75 = row.get("benchmark_最近一次投保距購買日_P75", np.nan)

        if pd.isna(x) or pd.isna(q25) or pd.isna(med) or pd.isna(q75):
            return np.nan
        if x <= q25:
            return 1.0
        elif x <= med:
            return 0.8
        elif x <= q75:
            return 0.5
        else:
            return 0.2

    df["rule_購買時機"] = df.apply(timing_score, axis=1)

    # 4. rule_消費力
    def spend_score(x, med, p75):
        if pd.isna(x) or pd.isna(med) or pd.isna(p75):
            return np.nan
        if x >= p75:
            return 1.0
        elif x >= med:
            return 0.7
        else:
            return 0.4

    spend_parts = []
    spend_pairs = [
        ("最大單張保單保費", "benchmark_最大單張保單保費_中位數", "benchmark_最大單張保單保費_P75"),
        ("平均每張保單保費", "benchmark_平均每張保單保費_中位數", "benchmark_平均每張保單保費_P75"),
        ("累計保單總保費", "benchmark_累計保單總保費_中位數", "benchmark_累計保單總保費_P75"),
    ]

    for raw_col, med_col, p75_col in spend_pairs:
        if raw_col in df.columns and med_col in df.columns and p75_col in df.columns:
            part_col = f"{raw_col}_消費力分數"
            df[part_col] = df.apply(lambda row: spend_score(row[raw_col], row[med_col], row[p75_col]), axis=1)
            spend_parts.append(part_col)

    df["rule_消費力"] = df[spend_parts].mean(axis=1, skipna=True) if spend_parts else np.nan

    # 5. bonus：前一張商品相似度
    if "benchmark_前一張商品主類別分數" in df.columns:
        s = df["benchmark_前一張商品主類別分數"].copy()
        if use_sqrt_scaling:
            s = np.sqrt(s)
        if s.notna().sum() > 0 and s.max() > s.min():
            df["rule_前一張商品相似度"] = (s - s.min()) / (s.max() - s.min())
        else:
            df["rule_前一張商品相似度"] = s
    else:
        df["rule_前一張商品相似度"] = np.nan

    # 6. bonus：前三張路徑相似度
    if "benchmark_前三張路徑分數" in df.columns:
        s = df["benchmark_前三張路徑分數"].copy()
        if use_sqrt_scaling:
            s = np.sqrt(s)
        if s.notna().sum() > 0 and s.max() > s.min():
            df["rule_前三張路徑相似度"] = (s - s.min()) / (s.max() - s.min())
        else:
            df["rule_前三張路徑相似度"] = s
    else:
        df["rule_前三張路徑相似度"] = np.nan

    # 7. bonus：購買頻率相似度
    def frequency_score(row):
        x = row.get("與前一張保單間距天數", np.nan)
        q25 = row.get("benchmark_最近間距_P25", np.nan)
        med = row.get("benchmark_最近間距_中位數", np.nan)
        q75 = row.get("benchmark_最近間距_P75", np.nan)

        if pd.isna(x) or pd.isna(q25) or pd.isna(med) or pd.isna(q75):
            return np.nan
        if x <= q25:
            return 1.0
        elif x <= med:
            return 0.8
        elif x <= q75:
            return 0.5
        else:
            return 0.2

    df["rule_購買頻率相似度"] = df.apply(frequency_score, axis=1)

    # 主分數
    numerator = pd.Series(0.0, index=df.index)
    denominator = pd.Series(0.0, index=df.index)

    for col, w in weight_config.items():
        if col in df.columns:
            has_value = df[col].notna().astype(float)
            numerator += df[col].fillna(0) * w
            denominator += has_value * w

    df["rule_score_main"] = np.where(
        denominator > 0,
        numerator / denominator,
        np.nan
    )

    # bonus 分數
    bonus_num = pd.Series(0.0, index=df.index)
    bonus_den = pd.Series(0.0, index=df.index)

    for col, w in bonus_weight_config.items():
        if col in df.columns:
            has_value = df[col].notna().astype(float)
            bonus_num += df[col].fillna(0) * w
            bonus_den += has_value * w

    df["rule_score_bonus"] = np.where(
        bonus_den > 0,
        bonus_num / bonus_den,
        0
    )

    # 最終總分
    # 主體 90%，bonus 10%
    df["rule_score"] = (
        0.90 * df["rule_score_main"].fillna(0) +
        0.10 * df["rule_score_bonus"].fillna(0)
    )

    df["rule_score_pct"] = df["rule_score"].rank(pct=True)
    df["rule_rank"] = df["rule_score"].rank(method="dense", ascending=False)

    sort_cols = ["rule_score"]
    ascending = [False]

    if "保單數" in df.columns:
        sort_cols.append("保單數")
        ascending.append(False)
    if "最大單張保單保費" in df.columns:
        sort_cols.append("最大單張保單保費")
        ascending.append(False)
    if "距離最近投保天數" in df.columns:
        sort_cols.append("距離最近投保天數")
        ascending.append(True)

    df = df.sort_values(sort_cols, ascending=ascending).reset_index(drop=True)
    df["overall_rule_rank"] = np.arange(1, len(df) + 1)

    return df


# %% 執行

import pandas as pd
import numpy as np
import re

# # 0. 時間權重小檢查
# tmp_check = benchmark_first_df.copy()
# tmp_check["首次躉繳投資型投保日"] = pd.to_datetime(tmp_check["首次躉繳投資型投保日"], errors="coerce")

# ref_date = tmp_check["首次躉繳投資型投保日"].max()
# tmp_check["距離基準日年數"] = (ref_date - tmp_check["首次躉繳投資型投保日"]).dt.days / 365.25
# tmp_check["時間權重"] = np.exp(-0.35 * tmp_check["距離基準日年數"])

# tmp_check[["首次躉繳投資型投保日", "距離基準日年數", "時間權重"]].sort_values("首次躉繳投資型投保日").tail(20)


# 1. 建立 benchmark profile  
benchmark_summary_df, seq_dist_df, product_rate_df, timing_profile_df, timing_by_seq_df, \
spending_profile_df, route_profile_df, frequency_profile_df = \
    build_benchmark_rule_profile(
        benchmark_snapshot_df=benchmark_snapshot_df,
        benchmark_policy_snapshot_df=benchmark_policy_snapshot_df,
        benchmark_first_df=benchmark_first_df
    )

# 2. 把 benchmark profile 套進 candidate
candidate_rule_base_df = apply_benchmark_profile_to_candidate(
    candidate_df=candidate_df,
    policy_df=policy_df,
    benchmark_summary_df=benchmark_summary_df,
    seq_dist_df=seq_dist_df,
    product_rate_df=product_rate_df,
    timing_profile_df=timing_profile_df,
    spending_profile_df=spending_profile_df,
    route_profile_df=route_profile_df,
    frequency_profile_df=frequency_profile_df
)

# 3. 算最終 Rule Score
rule_scored_df = build_rule_score_from_profile(candidate_rule_base_df)


# 檢查
# 確認 benchmark 路徑 / 頻率是否長得合理
route_profile_df.head(20)
frequency_profile_df.head(20)

# # candidate 套入後是否有值
# candidate_rule_base_df[
#     [
#         "被保人身分證字號",
#         "benchmark_前一張商品主類別分數",
#         "benchmark_前三張路徑分數",
#         "benchmark_最近間距_中位數",
#         "與前一張保單間距天數"
#     ]
# ].head(20)

# # rule score 是否合理
# top = rule_scored_df[
#     [
#         "被保人身分證字號",
#         "rule_score_main",
#         "rule_score_bonus",
#         "rule_score",
#         "overall_rule_rank"
#     ]
# ].head(30)

import matplotlib.pyplot as plt
plt.rc('font', family = 'Microsoft JhengHei')
plt.rcParams['axes.unicode_minus'] = False

data = rule_scored_df["rule_score"].dropna()

plt.figure(figsize=(8, 5))

counts, bins, patches = plt.hist(data, bins=50, alpha=0.6)

# 標記每個 bin 的人數
for i in range(len(counts)):
    if counts[i] > 0:
        x = (bins[i] + bins[i+1]) / 2
        y = counts[i]
        plt.text(x, y, int(counts[i]), ha='center', va='bottom', fontsize=8)

# KDE
data.plot(kind="kde")

plt.title("Rule Score Distribution + KDE")
plt.xlabel("Rule Score")

plt.show()

rule_scored_df.boxplot(column="保單數", by="score_group")

rule_scored_df["rule_score"].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99])

# 三群人數
group_count_df = (
    rule_scored_df["score_group"]
    .value_counts()
    .sort_index()
    .reset_index()
)

group_count_df.columns = ["score_group", "人數"]

# 加上比例
group_count_df["比例"] = group_count_df["人數"] / group_count_df["人數"].sum()

group_count_df


rule_scored_df = rule_scored_df.sort_values("rule_score")

rule_scored_df["cum_pct"] = (
    np.arange(len(rule_scored_df)) / len(rule_scored_df)
)

plt.figure(figsize=(8,5))
plt.plot(rule_scored_df["rule_score"], rule_scored_df["cum_pct"])

plt.xlabel("Rule Score")
plt.ylabel("Cumulative %")
plt.title("Cumulative Distribution")

plt.show()





rule_scored_df["score_group"] = pd.qcut(
    rule_scored_df["rule_score"],
    q=[0, 0.5, 0.8, 1.0],
    labels=["Bottom 50%", "Middle 30%", "Top 20%"]
)

rule_scored_df.groupby("score_group")["保單數"].mean()

benchmark_seq_mean = benchmark_summary_df.loc[
    benchmark_summary_df["欄位"] == "保單數",
    "加權平均數"
].values[0]

compare_df = rule_scored_df.groupby("score_group").agg({
    "保單數": "mean",
    "最大單張保單保費": "mean"
}).reset_index()

compare_df["保單數_距離"] = abs(compare_df["保單數"] - benchmark_seq_mean)
compare_df.head()



top_df = rule_scored_df[rule_scored_df["score_group"] == "Top 20%"].copy()

top_seq_dist_df = (
    top_df["保單數"]
    .value_counts(normalize=True)
    .sort_index()
    .reset_index()
)

top_seq_dist_df.columns = ["保單數", "比例"]

top_product_df = (
    top_df["前一張商品主類別"]
    .value_counts(normalize=True)
    .reset_index()
)

top_product_df.columns = ["商品類別", "比例"]

top_timing_summary = top_df[[
    "距離最近投保天數"
]].describe()


seq_score_analysis = (
    rule_scored_df
    .groupby("保單數")
    .agg(
        人數=("被保人身分證字號", "count"),
        平均分數=("rule_score", "mean"),
        中位數分數=("rule_score", "median")
    )
    .reset_index()
    .sort_values("保單數")
)




benchmark_snapshot_df["首次躉繳投資型投保日"] = pd.to_datetime(benchmark_snapshot_df["首次躉繳投資型投保日"], errors="coerce")

benchmark_snapshot_df["購買年份"] = benchmark_snapshot_df["首次躉繳投資型投保日"].dt.year


timing_year_profile = (
    benchmark_snapshot_df.groupby("購買年份")["距離首次買躉繳投資型前最近一次投保天數"]
    .agg(
        客戶數="count",
        平均="mean",
        中位數="median",
        P25=lambda x: x.quantile(0.25),
        P75=lambda x: x.quantile(0.75),
        最大值="max"
    )
    .reset_index()
)


timing_profile = (
    benchmark_snapshot_df["距離首次買躉繳投資型前最近一次投保天數"]
    .agg(
        客戶數="count",
        平均="mean",
        中位數="median",
        P25=lambda x: x.quantile(0.25),
        P75=lambda x: x.quantile(0.75),
        最大值="max"
    )
    .reset_index()
)


