import pandas as pd
import numpy as np

def get_output_schema():
    return pd.DataFrame({
        "經紀人 業代": prep_string(),
        "check_week": prep_string(),
        "目前開放批次": prep_string(),
        "本週檢查批次": prep_string(),
        "該批客戶數": prep_string(),
        "已拜訪數": prep_string(),
        "該批拜訪率": prep_string(),
        "是否達標": prep_string(),
        "下週開放批次": prep_string(),
        "總批數": prep_string(),
        "批次進度": prep_string()
    })

def calc_release_status_string(df):
    progress = df.copy()

    required_cols = [
        "是否達標",
        "該批拜訪率",
        "已拜訪數",
        "經紀人 業代",
        "Release Batch",
        "check_week",
        "該批客戶數"
    ]

    missing_cols = [c for c in required_cols if c not in progress.columns]
    if missing_cols:
        raise ValueError(f"缺少必要欄位: {missing_cols}")

    progress = progress.rename(columns={
        "經紀人 業代": "agent_id",
        "Release Batch": "batch_no",
        "該批客戶數": "batch_customer_cnt",
        "已拜訪數": "batch_visited_cnt",
        "該批拜訪率": "visit_rate",
        "是否達標": "pass_flag"
    })

    # 全部保守處理
    progress["agent_id"] = progress["agent_id"].astype(str).str.strip()
    progress["check_week_text"] = progress["check_week"].astype(str).str.strip()

    progress["batch_no"] = pd.to_numeric(progress["batch_no"], errors="coerce")
    progress["batch_customer_cnt"] = pd.to_numeric(progress["batch_customer_cnt"], errors="coerce").fillna(0)
    progress["batch_visited_cnt"] = pd.to_numeric(progress["batch_visited_cnt"], errors="coerce").fillna(0)
    progress["visit_rate"] = pd.to_numeric(progress["visit_rate"], errors="coerce").fillna(0)
    progress["pass_flag"] = pd.to_numeric(progress["pass_flag"], errors="coerce").fillna(0).astype(int)

    # 用字串排序週別前，先盡量轉 datetime；失敗也不直接炸
    progress["check_week_dt"] = pd.to_datetime(progress["check_week_text"], errors="coerce")

    # 若轉不出日期，就用原字串；但至少不能是空值
    progress = progress[
        progress["agent_id"].notna() &
        progress["agent_id"].ne("") &
        progress["batch_no"].notna() &
        progress["check_week_text"].notna() &
        progress["check_week_text"].ne("")
    ].copy()

    if progress.empty:
        return pd.DataFrame(columns=[
            "經紀人 業代",
            "check_week",
            "目前開放批次",
            "本週檢查批次",
            "該批客戶數",
            "已拜訪數",
            "該批拜訪率",
            "是否達標",
            "下週開放批次",
            "總批數",
            "批次進度"
        ])

    # 優先用 datetime 排序，失敗的排後面
    progress = progress.sort_values(
        by=["agent_id", "check_week_dt", "batch_no"],
        na_position="last"
    ).copy()

    agent_total_batches = (
        progress.groupby("agent_id", as_index=False)["batch_no"]
        .max()
        .rename(columns={"batch_no": "total_batches"})
    )

    all_weeks = list(progress["check_week_text"].drop_duplicates())

    result_rows = []

    for _, row in agent_total_batches.iterrows():
        agent_id = row["agent_id"]
        total_batches = int(row["total_batches"])
        agent_progress = progress[progress["agent_id"] == agent_id].copy()

        current_open_batch = 1

        for wk in all_weeks:
            wk_rows = agent_progress[agent_progress["check_week_text"] == wk].copy()
            checked_batch_no = current_open_batch
            target = wk_rows[wk_rows["batch_no"] == checked_batch_no].copy()

            if len(target) > 0:
                batch_customer_cnt = int(target["batch_customer_cnt"].iloc[0])
                batch_visited_cnt = int(target["batch_visited_cnt"].iloc[0])
                visit_rate = float(target["visit_rate"].iloc[0])
                pass_flag = int(target["pass_flag"].iloc[0])
            else:
                batch_customer_cnt = 0
                batch_visited_cnt = 0
                visit_rate = 0.0
                pass_flag = 0

            if (pass_flag == 1) and (current_open_batch < total_batches):
                next_open_batch = current_open_batch + 1
            else:
                next_open_batch = current_open_batch

            result_rows.append({
                "經紀人 業代": str(agent_id),
                "check_week": str(wk),
                "目前開放批次": str(int(current_open_batch)),
                "本週檢查批次": str(int(checked_batch_no)),
                "該批客戶數": str(int(batch_customer_cnt)),
                "已拜訪數": str(int(batch_visited_cnt)),
                "該批拜訪率": str(round(visit_rate, 4)),
                "是否達標": str(int(pass_flag)),
                "下週開放批次": str(int(next_open_batch)),
                "總批數": str(int(total_batches)),
                "批次進度": f"{int(current_open_batch)}/{int(total_batches)}"
            })

            current_open_batch = next_open_batch

    result = pd.DataFrame(result_rows)

    return result[[
        "經紀人 業代",
        "check_week",
        "目前開放批次",
        "本週檢查批次",
        "該批客戶數",
        "已拜訪數",
        "該批拜訪率",
        "是否達標",
        "下週開放批次",
        "總批數",
        "批次進度"
    ]]