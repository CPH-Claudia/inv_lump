# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 16:32:43 2026

@author: Z01788
"""
import pandas as pd
import numpy as np

# %% 建立轉換率分箱函式：讓每箱樣本數盡量接近
# 套用分箱
def apply_bins(series: pd.Series, bin_edges: list, right: bool = True):
    """
    將數值欄位依指定邊界分箱。
    """
    if bin_edges is None:
        return pd.Series(["ALL"] * len(series), index=series.index, dtype="string")

    s = pd.to_numeric(series, errors="coerce")

    binned = pd.cut(
        s,
        bins=bin_edges,
        include_lowest=True,
        right=right
    )

    return binned.astype("string").fillna("Missing")

def make_hybrid_bins_from_combined(
    benchmark_df: pd.DataFrame,
    candidate_df: pd.DataFrame,
    col: str,
    q: int = 5,
    low_value_config: dict = None,
    duplicates: str = "drop"
):
    """
    Hybrid 分箱：
    1. 若欄位在 low_value_config 中，先保留指定低值各自成箱
    2. 剩下的值再用 qcut 做等量分箱
    3. 其他欄位則直接 qcut
    """
    if low_value_config is None:
        low_value_config = {}

    s = pd.concat([
        benchmark_df[col],
        candidate_df[col]
    ], axis=0).dropna()

    s = pd.to_numeric(s, errors="coerce").dropna()

    if s.empty:
        return None

    # ===== 有指定低值保留 =====
    if col in low_value_config:
        preserve_vals = sorted(set(low_value_config[col]))
        preserve_vals = [v for v in preserve_vals if (s == v).any()]

        if len(preserve_vals) == 0:
            # fallback: 一般 qcut
            try:
                _, bin_edges = pd.qcut(s, q=q, retbins=True, duplicates=duplicates)
                bin_edges = np.unique(bin_edges)
                if len(bin_edges) <= 2:
                    return None
                bin_edges[0] = -np.inf
                bin_edges[-1] = np.inf
                return bin_edges.tolist()
            except Exception:
                return None

        max_preserve = max(preserve_vals)
        tail = s[s > max_preserve]

        edges = [-np.inf]
        for v in preserve_vals:
            edges.append(v)

        if len(tail) > 0 and tail.nunique() > 1:
            tail_q = max(q - len(preserve_vals), 1)
            try:
                _, tail_edges = pd.qcut(
                    tail,
                    q=tail_q,
                    retbins=True,
                    duplicates=duplicates
                )
                tail_edges = np.unique(tail_edges)

                for e in tail_edges:
                    if e > max_preserve:
                        edges.append(e)
            except Exception:
                pass

        edges.append(np.inf)
        edges = sorted(set(edges))

        if len(edges) <= 2:
            return None

        return edges

    # ===== 一般欄位：直接 qcut =====
    try:
        _, bin_edges = pd.qcut(s, q=q, retbins=True, duplicates=duplicates)
        bin_edges = np.unique(bin_edges)

        if len(bin_edges) <= 2:
            return None

        bin_edges[0] = -np.inf
        bin_edges[-1] = np.inf
        return bin_edges.tolist()

    except Exception:
        return None


# %% 建立每個欄位的轉換率 score table
def build_conversion_rate_score_table(
    benchmark_df: pd.DataFrame,
    candidate_df: pd.DataFrame,
    col: str,
    q: int = 5,
    min_total_count: int = 30,
    smoothing_k: int = 100,
    low_value_config: dict = None
):
    """
    單一欄位：
    1. 分箱
    2. 算每箱 benchmark / candidate 數量
    3. 算轉換率
    4. 平滑
    5. 轉成 0-100 score
    """

    bin_edges = make_hybrid_bins_from_combined(
        benchmark_df=benchmark_df,
        candidate_df=candidate_df,
        col=col,
        q=q,
        low_value_config=low_value_config
    )

    bench = benchmark_df[[col]].copy()
    cand = candidate_df[[col]].copy()

    bench["bin"] = apply_bins(bench[col], bin_edges)
    cand["bin"] = apply_bins(cand[col], bin_edges)

    bench["bin"] = bench["bin"].astype("string")
    cand["bin"] = cand["bin"].astype("string")

    bench_dist = bench["bin"].value_counts(dropna=False).rename("benchmark_cnt").reset_index()
    bench_dist.columns = ["bin", "benchmark_cnt"]

    cand_dist = cand["bin"].value_counts(dropna=False).rename("candidate_cnt").reset_index()
    cand_dist.columns = ["bin", "candidate_cnt"]

    score_table = bench_dist.merge(cand_dist, on="bin", how="outer")
    score_table["benchmark_cnt"] = score_table["benchmark_cnt"].fillna(0).astype(int)
    score_table["candidate_cnt"] = score_table["candidate_cnt"].fillna(0).astype(int)
    score_table["bin"] = score_table["bin"].astype("string").fillna("Missing")

    score_table["total_cnt"] = score_table["benchmark_cnt"] + score_table["candidate_cnt"]

    # 全體 baseline conversion rate
    global_benchmark = len(benchmark_df)
    global_candidate = len(candidate_df)
    global_rate = global_benchmark / (global_benchmark + global_candidate)

    # 原始轉換率
    score_table["raw_conversion_rate"] = np.where(
        score_table["total_cnt"] > 0,
        score_table["benchmark_cnt"] / score_table["total_cnt"],
        np.nan
    )

    # 平滑後轉換率
    score_table["smoothed_conversion_rate"] = (
        (score_table["benchmark_cnt"] + smoothing_k * global_rate) /
        (score_table["total_cnt"] + smoothing_k)
    )

    # 樣本數門檻
    score_table["usable"] = score_table["total_cnt"] >= min_total_count

    score_table["final_conversion_rate"] = np.where(
        score_table["usable"],
        score_table["smoothed_conversion_rate"],
        global_rate
    )

    # 轉成 0-100 score
    min_rate = score_table["final_conversion_rate"].min()
    max_rate = score_table["final_conversion_rate"].max()

    if max_rate > min_rate:
        score_table["score"] = 100 * (
            (score_table["final_conversion_rate"] - min_rate) / (max_rate - min_rate)
        )
    else:
        score_table["score"] = 50.0

    score_table["score"] = score_table["score"].round(1)
    score_table["feature"] = col

    # 依箱下界排序
    def _extract_left_boundary(bin_str):
        try:
            s = str(bin_str).replace("[", "(").replace("]", ")")
            left = s.split(",")[0].replace("(", "").strip()
            if left == "-inf":
                return -np.inf
            return float(left)
        except Exception:
            return np.inf

    score_table["_sort_key"] = score_table["bin"].apply(_extract_left_boundary)
    score_table = score_table.sort_values("_sort_key").drop(columns="_sort_key").reset_index(drop=True)

    return score_table, bin_edges


# %% 批次建立多個欄位的 score table
def build_rule_score_tables(
    benchmark_df: pd.DataFrame,
    candidate_df: pd.DataFrame,
    feature_config: dict,
    q: int = 5,
    min_total_count: int = 30,
    smoothing_k: int = 100,
    low_value_config: dict = None
):
    score_tables = {}
    bin_edges_dict = {}

    for col in feature_config.keys():
        score_table, bin_edges = build_conversion_rate_score_table(
            benchmark_df=benchmark_df,
            candidate_df=candidate_df,
            col=col,
            q=q,
            min_total_count=min_total_count,
            smoothing_k=smoothing_k,
            low_value_config=low_value_config
        )
        score_tables[col] = score_table
        bin_edges_dict[col] = bin_edges

    return score_tables, bin_edges_dict


# %% 將 score table 套回 profile，產生 rule score
def build_rule_score_from_profile(
    profile_df: pd.DataFrame,
    feature_config: dict,
    group_weights: dict,
    score_tables: dict,
    bin_edges_dict: dict,
    id_col: str = "被保人身分證字號"
):
    df = profile_df.copy()

    if id_col not in df.columns:
        raise ValueError(f"profile_df 缺少必要欄位: {id_col}")

    missing_cols = [c for c in feature_config.keys() if c not in df.columns]
    if missing_cols:
        raise ValueError(f"profile_df 缺少 rule score 所需欄位: {missing_cols}")

    group_feature_map = {}

    # 1) feature-level score
    for col, cfg in feature_config.items():
        bin_col = f"{col}_bin"
        score_col = f"score_{col}"

        df[bin_col] = apply_bins(df[col], bin_edges_dict[col])

        mapping = score_tables[col].set_index("bin")["score"].to_dict()
        df[score_col] = df[bin_col].map(mapping).fillna(50)

        group_name = cfg["group"]
        group_feature_map.setdefault(group_name, []).append(
            (col, score_col, cfg["feature_weight"])
        )

    # 2) group-level score
    for group_name, items in group_feature_map.items():
        score_cols = [x[1] for x in items]
        weights = np.array([x[2] for x in items], dtype=float)
        weights = weights / weights.sum()

        df[f"{group_name}_score"] = 0.0
        for (_, score_col, _), w in zip(items, weights):
            df[f"{group_name}_score"] += df[score_col] * w

        df[f"{group_name}_score"] = df[f"{group_name}_score"].round(2)

    # 3) final rule score
    valid_groups = [g for g in group_weights if f"{g}_score" in df.columns]
    total_group_weight = sum(group_weights[g] for g in valid_groups)

    df["rule_score"] = 0.0
    for g in valid_groups:
        w = group_weights[g] / total_group_weight
        df["rule_score"] += df[f"{g}_score"] * w

    df["rule_score"] = df["rule_score"].round(2)

    return df

# %% 主程式
def run_rule_scoring_pipeline(
    customer_df: pd.DataFrame,
    benchmark_snapshot_df: pd.DataFrame,
    feature_config: dict,
    group_weights: dict,
    low_value_config: dict = None,
    q: int = 5,
    min_total_count: int = 30,
    smoothing_k: int = 100,
    id_col: str = "被保人身分證字號"
):
    """
    完整 rule scoring pipeline

    流程：
    1. 從 customer_df 建 candidate_df
    2. 檢查 feature 是否存在
    3. 建 score_tables
    4. 計算 candidate rule_score
    """

    # 1) candidate pool
    candidate_df, funnel_df = build_candidate_pool(customer_df)

    benchmark_profile_for_rule = benchmark_snapshot_df.copy()
    candidate_profile_for_rule = candidate_df.copy()

    # 2) 檢查欄位
    validate_result = validate_rule_features(
        feature_config=feature_config,
        benchmark_df=benchmark_profile_for_rule,
        candidate_df=candidate_profile_for_rule
    )

    if validate_result["missing_in_benchmark"] or validate_result["missing_in_candidate"]:
        raise ValueError(
            f"feature_config 欄位不齊。\n"
            f"missing_in_benchmark={validate_result['missing_in_benchmark']}\n"
            f"missing_in_candidate={validate_result['missing_in_candidate']}"
        )

    # 3) score tables
    score_tables, bin_edges_dict = build_rule_score_tables(
        benchmark_df=benchmark_profile_for_rule,
        candidate_df=candidate_profile_for_rule,
        feature_config=feature_config,
        q=q,
        min_total_count=min_total_count,
        smoothing_k=smoothing_k,
        low_value_config=low_value_config
    )

    # 4) candidate scoring
    candidate_rule_scored_df = build_rule_score_from_profile(
        profile_df=candidate_profile_for_rule,
        feature_config=feature_config,
        group_weights=group_weights,
        score_tables=score_tables,
        bin_edges_dict=bin_edges_dict,
        id_col=id_col
    )

    # 5) 排序
    candidate_rule_scored_df = candidate_rule_scored_df.sort_values(
        "rule_score", ascending=False
    ).reset_index(drop=True)

    return {
        "candidate_df": candidate_df,
        "funnel_df": funnel_df,
        "score_tables": score_tables,
        "bin_edges_dict": bin_edges_dict,
        "candidate_rule_scored_df": candidate_rule_scored_df
    }

# %% 執行
rule_pipeline_result = run_rule_scoring_pipeline(
    customer_df=customer_df,
    benchmark_snapshot_df=benchmark_snapshot_df,
    feature_config=feature_config,
    group_weights=group_weights,
    low_value_config=COUNT_LIKE_BIN_CONFIG,
    q=5,
    min_total_count=30,
    smoothing_k=100,
    id_col="被保人身分證字號"
)

candidate_df = rule_pipeline_result["candidate_df"]
funnel_df = rule_pipeline_result["funnel_df"]
score_tables = rule_pipeline_result["score_tables"]
bin_edges_dict = rule_pipeline_result["bin_edges_dict"]
candidate_rule_scored_df = rule_pipeline_result["candidate_rule_scored_df"]

# %% 看細節
# 看 top candidate
candidate_rule_scored_df[
    [
        "被保人身分證字號",
        "rule_score",
        "壽險參與度_score",
        "保費能力_score",
        "關係深度_score",
        "FYC結構_score",
        "近期行為_score"
    ]
].head(20)

# 看某欄位分箱表
壽險保單數 = score_tables["壽險保單數"]
保單數 = score_tables["保單數"]
score_tables["近1年保單數"]

import matplotlib.pyplot as plt
plt.rc('font', family = 'Microsoft JhengHei')
plt.rcParams['axes.unicode_minus'] = False

data = candidate_rule_scored_df["rule_score"].dropna()
plt.figure(figsize=(8, 5))
counts, bins, patches = plt.hist(data, bins=50, alpha=0.6)


# %% 輸出
with pd.ExcelWriter("rule_scoring_output.xlsx", engine="openpyxl") as writer:
    funnel_df.to_excel(writer, sheet_name="funnel", index=False)
    candidate_rule_scored_df.to_excel(writer, sheet_name="candidate_rule_score", index=False)

    for feature_name, st in score_tables.items():
        sheet_name = f"score_{feature_name}"[:31]  # Excel sheet name 最長 31 字
        st.to_excel(writer, sheet_name=sheet_name, index=False)

# %% 建 candidate pool
def build_candidate_pool(
    customer_df: pd.DataFrame,
    require_positive_policy_cnt: bool = True
):
    df = customer_df.copy()
    df.columns = df.columns.str.strip()

    required_cols = ["被保人身分證字號", "是否曾買過躉繳投資型"]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"customer_df 缺少必要欄位: {missing_cols}")

    df["是否曾買過躉繳投資型"] = pd.to_numeric(
        df["是否曾買過躉繳投資型"], errors="coerce"
    )

    funnel_records = []
    funnel_records.append(["全部客戶", len(df)])

    candidate_df = df[df["是否曾買過躉繳投資型"] == 0].copy()
    funnel_records.append(["未買躉繳投資型", len(candidate_df)])

    if require_positive_policy_cnt and "保單數" in candidate_df.columns:
        before_cnt = len(candidate_df)
        candidate_df = candidate_df[candidate_df["保單數"].fillna(0) > 0].copy()
        funnel_records.append(["保單數 > 0", len(candidate_df)])
        funnel_records.append(["因保單數<=0被排除", before_cnt - len(candidate_df)])

    candidate_df["candidate標記"] = 1

    funnel_df = pd.DataFrame(funnel_records, columns=["漏斗階段", "客戶數"])

    return candidate_df, funnel_df


# %% 執行
benchmark_profile_for_rule = benchmark_snapshot_df.copy()
candidate_profile_for_rule = candidate_df.copy()

# =========================
# A. rule feature 設定
# =========================
feature_config = {
    # 1) 壽險參與度
    "壽險保單數": {
        "group": "壽險參與度",
        "feature_weight": 0.55,
    },
    "壽險保單占比": {
        "group": "壽險參與度",
        "feature_weight": 0.45,
    },

    # 2) 保費能力
    "累計壽險保單總保費": {
        "group": "保費能力",
        "feature_weight": 0.45,
    },
    "壽險保費占比": {
        "group": "保費能力",
        "feature_weight": 0.30,
    },
    "累計保單總保費": {
        "group": "保費能力",
        "feature_weight": 0.25,
    },

    # 3) 關係深度
    "保單數": {
        "group": "關係深度",
        "feature_weight": 0.70,
    },
    "產險保單數": {
        "group": "關係深度",
        "feature_weight": 0.30,
    },

    # 4) FYC結構
    "累計壽險保單總繳款FYC": {
        "group": "FYC結構",
        "feature_weight": 0.60,
    },
    "壽險繳款FYC占比": {
        "group": "FYC結構",
        "feature_weight": 0.40,
    },

    # 5) 近期行為
    "近1年保單數": {
        "group": "近期行為",
        "feature_weight": 1.00,
    },
}

# =========================
# B. 群組權重
# =========================
group_weights = {
    "壽險參與度": 0.35,
    "保費能力": 0.25,
    "關係深度": 0.15,
    "FYC結構": 0.10,
    "近期行為": 0.15,
}

# =========================
# C. 件數型欄位的 hybrid 分箱
#    保留低值單獨成箱，其餘尾端再 qcut
# =========================
COUNT_LIKE_BIN_CONFIG = {
    "壽險保單數": [0, 1, 3],
    "保單數": [0, 1],
    "產險保單數": [0, 1],
    "近1年保單數": [0, 1, 2],
}


# 建立所有欄位的 score table
score_tables, bin_edges_dict = build_rule_score_tables(
    benchmark_df=benchmark_profile_for_rule,
    candidate_df=candidate_profile_for_rule,
    feature_config=feature_config,
    q=5,                 # 先切 5 箱
    min_total_count=30,  # 每箱至少 30 筆
    smoothing_k=100      # 平滑強度
)

# 單獨開一張表檢查
壽險保單數 = score_tables["壽險保單數"] 

# 把 score table 套回 candidate，生成 rule score
candidate_rule_scored_df = build_rule_score_from_profile(
    profile_df=candidate_profile_for_rule,
    feature_config=feature_config,
    score_tables=score_tables,
    bin_edges_dict=bin_edges_dict,
    id_col="被保人身分證字號"
)

# 看結果
candidate_rule_scored_df[
    [
        "被保人身分證字號",
        "rule_score",
        "關係深度_score",
        "保費能力_score",
        "FYC結構_score",
        "商品成熟度_score",
        "近期活躍度_score"
    ]
].sort_values("rule_score", ascending=False).head(20)

# 檢查
# 檢查每個欄位的 score table
score_tables["保單數"]
score_tables["壽險保單數"]
score_tables["累計壽險保單總保費"]

# 看 top rule score 客群輪廓
candidate_rule_scored_df["rule_score_rank_pct"] = (
    candidate_rule_scored_df["rule_score"].rank(pct=True)
)

top20 = candidate_rule_scored_df[
    candidate_rule_scored_df["rule_score_rank_pct"] >= 0.8
]

top20[[
    "保單數",
    "壽險保單數",
    "壽險保單占比",
    "累計保單總保費",
    "累計壽險保單總保費",
    "壽險保費占比"
]].describe()







COUNT_LIKE_BIN_CONFIG = {
    "壽險保單數": [0, 1, 3],
    "保單數": [0, 1],
    "近1年保單數": [0, 1, 2],
}


def make_hybrid_bins_from_combined(
    benchmark_df: pd.DataFrame,
    candidate_df: pd.DataFrame,
    col: str,
    q: int = 5,
    low_value_config: dict = None,
    duplicates: str = "drop"
):
    """
    Hybrid 分箱：
    1. 若欄位在 low_value_config 中，先保留指定低值各自成箱
    2. 剩下的值再用 qcut 做等量分箱
    3. 其他欄位則直接 qcut

    回傳
    ----------
    bin_edges : list 或 None
        可供 pd.cut 使用的邊界
    """
    if low_value_config is None:
        low_value_config = {}

    s = pd.concat([
        benchmark_df[col],
        candidate_df[col]
    ], axis=0).dropna()

    if s.empty:
        return None

    # 統一轉 numeric（如果本來已是 numeric 不影響）
    s = pd.to_numeric(s, errors="coerce").dropna()

    if s.empty:
        return None

    # ===== Case 1: 有指定低值保留 =====
    if col in low_value_config:
        preserve_vals = sorted(set(low_value_config[col]))

        # 只保留資料中真的存在的值
        preserve_vals = [v for v in preserve_vals if (s == v).any()]

        # 若沒有任何 preserve 值存在，就退回一般 qcut
        if len(preserve_vals) == 0:
            try:
                _, bin_edges = pd.qcut(s, q=q, retbins=True, duplicates=duplicates)
                bin_edges = np.unique(bin_edges)
                if len(bin_edges) <= 2:
                    return None
                return bin_edges.tolist()
            except Exception:
                return None

        max_preserve = max(preserve_vals)

        # 剩餘值：大於 max_preserve 的部分
        tail = s[s > max_preserve]

        # 建立 cut 邊界
        edges = [-np.inf]

        # 讓 preserve 值各自成箱
        # 例如 [0,1] -> (-inf,0], (0,1]
        for v in preserve_vals:
            edges.append(v)

        # tail 再做 qcut
        if len(tail) > 0 and tail.nunique() > 1:
            # 剩餘要切幾箱：總箱數 q 減去已固定箱數
            tail_q = max(q - len(preserve_vals), 1)

            try:
                _, tail_edges = pd.qcut(
                    tail,
                    q=tail_q,
                    retbins=True,
                    duplicates=duplicates
                )
                tail_edges = np.unique(tail_edges)

                # tail_edges 第一個會等於 tail.min()，這裡不要重複接到 preserve 邊界
                for e in tail_edges:
                    if e > max_preserve:
                        edges.append(e)

            except Exception:
                # tail 無法 qcut 時，直接把 tail 併成一大箱
                pass

        edges.append(np.inf)

        # 去重、排序
        edges = sorted(set(edges))

        # 至少要有 3 個邊界才形成 2 箱
        if len(edges) <= 2:
            return None

        return edges

    # ===== Case 2: 一般欄位直接 qcut =====
    try:
        _, bin_edges = pd.qcut(s, q=q, retbins=True, duplicates=duplicates)
        bin_edges = np.unique(bin_edges)

        if len(bin_edges) <= 2:
            return None

        # 為了後續 pd.cut 較穩，補成開放邊界
        bin_edges[0] = -np.inf
        bin_edges[-1] = np.inf

        return bin_edges.tolist()

    except Exception:
        return None


def apply_bins(series: pd.Series, bin_edges: list, right: bool = True):
    """
    將數值欄位依指定邊界分箱。
    """
    if bin_edges is None:
        return pd.Series(["ALL"] * len(series), index=series.index, dtype="string")

    s = pd.to_numeric(series, errors="coerce")

    binned = pd.cut(
        s,
        bins=bin_edges,
        include_lowest=True,
        right=right
    )

    return binned.astype("string").fillna("Missing")


def build_conversion_rate_score_table(
    benchmark_df: pd.DataFrame,
    candidate_df: pd.DataFrame,
    col: str,
    q: int = 5,
    min_total_count: int = 30,
    smoothing_k: int = 100,
    low_value_config: dict = None
):
    """
    針對單一欄位：
    1. Hybrid 分箱（低值保留 + 尾端 qcut）
    2. 算每箱 benchmark / candidate 數量
    3. 算每箱原始轉換率
    4. 做平滑
    5. 轉成 0-100 score
    """

    # 1) 建立 hybrid 分箱邊界
    bin_edges = make_hybrid_bins_from_combined(
        benchmark_df=benchmark_df,
        candidate_df=candidate_df,
        col=col,
        q=q,
        low_value_config=low_value_config
    )

    bench = benchmark_df[[col]].copy()
    cand = candidate_df[[col]].copy()

    bench["bin"] = apply_bins(bench[col], bin_edges)
    cand["bin"] = apply_bins(cand[col], bin_edges)

    bench["bin"] = bench["bin"].astype("string")
    cand["bin"] = cand["bin"].astype("string")

    # 2) 各箱數量
    bench_dist = bench["bin"].value_counts(dropna=False).rename("benchmark_cnt").reset_index()
    bench_dist.columns = ["bin", "benchmark_cnt"]

    cand_dist = cand["bin"].value_counts(dropna=False).rename("candidate_cnt").reset_index()
    cand_dist.columns = ["bin", "candidate_cnt"]

    score_table = bench_dist.merge(cand_dist, on="bin", how="outer")
    score_table["benchmark_cnt"] = score_table["benchmark_cnt"].fillna(0).astype(int)
    score_table["candidate_cnt"] = score_table["candidate_cnt"].fillna(0).astype(int)
    score_table["bin"] = score_table["bin"].astype("string").fillna("Missing")

    score_table["total_cnt"] = score_table["benchmark_cnt"] + score_table["candidate_cnt"]

    # 3) 全體 baseline conversion rate
    global_benchmark = len(benchmark_df)
    global_candidate = len(candidate_df)
    global_rate = global_benchmark / (global_benchmark + global_candidate)

    # 4) 原始轉換率
    score_table["raw_conversion_rate"] = np.where(
        score_table["total_cnt"] > 0,
        score_table["benchmark_cnt"] / score_table["total_cnt"],
        np.nan
    )

    # 5) 平滑後轉換率
    score_table["smoothed_conversion_rate"] = (
        (score_table["benchmark_cnt"] + smoothing_k * global_rate) /
        (score_table["total_cnt"] + smoothing_k)
    )

    # 6) 樣本數門檻
    score_table["usable"] = score_table["total_cnt"] >= min_total_count

    score_table["final_conversion_rate"] = np.where(
        score_table["usable"],
        score_table["smoothed_conversion_rate"],
        global_rate
    )

    # 7) 轉成 0-100 score
    min_rate = score_table["final_conversion_rate"].min()
    max_rate = score_table["final_conversion_rate"].max()

    if max_rate > min_rate:
        score_table["score"] = 100 * (
            (score_table["final_conversion_rate"] - min_rate) / (max_rate - min_rate)
        )
    else:
        score_table["score"] = 50.0

    score_table["score"] = score_table["score"].round(1)
    score_table["feature"] = col

    # 依箱下界排序，比較好看
    def _extract_left_boundary(bin_str):
        try:
            s = str(bin_str).replace("[", "(").replace("]", ")")
            left = s.split(",")[0].replace("(", "").strip()
            if left == "-inf":
                return -np.inf
            return float(left)
        except Exception:
            return np.inf

    score_table["_sort_key"] = score_table["bin"].apply(_extract_left_boundary)
    score_table = score_table.sort_values("_sort_key").drop(columns="_sort_key").reset_index(drop=True)

    return score_table, bin_edges

def build_rule_score_tables(
    benchmark_df: pd.DataFrame,
    candidate_df: pd.DataFrame,
    feature_config: dict,
    q: int = 5,
    min_total_count: int = 30,
    smoothing_k: int = 100,
    low_value_config: dict = None
):
    """
    批次建立多個欄位的 score table 與 bin edges。
    """
    score_tables = {}
    bin_edges_dict = {}

    for col in feature_config.keys():
        score_table, bin_edges = build_conversion_rate_score_table(
            benchmark_df=benchmark_df,
            candidate_df=candidate_df,
            col=col,
            q=q,
            min_total_count=min_total_count,
            smoothing_k=smoothing_k,
            low_value_config=low_value_config
        )
        score_tables[col] = score_table
        bin_edges_dict[col] = bin_edges

    return score_tables, bin_edges_dict

score_tables, bin_edges_dict = build_rule_score_tables(
    benchmark_df=benchmark_profile_for_rule,
    candidate_df=candidate_profile_for_rule,
    feature_config=feature_config,
    q=5,
    min_total_count=30,
    smoothing_k=100,
    low_value_config=COUNT_LIKE_BIN_CONFIG
)


保單數 = score_tables["保單數"]
近1年保單數 = score_tables["近1年保單數"]
主約投資型保單數_依主類別 = score_tables["主約投資型保單數_依主類別"]
