"""Shared data loading and cleaning for app.py (Streamlit) and api.py (FastAPI).

Looks for loans.pkl, customers.pkl and payments.pkl in ./data/raw, ./Data/raw
or the current folder.
"""
import os

import pandas as pd

TABLES = ["loans", "customers", "payments"]
CANDIDATE_DIRS = ["data/raw", "Data/raw", "data", "Data", "."]
RECENT_FROM = "2026-05-01"  # processor change / deterioration starts here


def find_data_dir():
    for folder in CANDIDATE_DIRS:
        if all(os.path.exists(os.path.join(folder, f"{t}.pkl")) for t in TABLES):
            return folder
    return None


def _action(row):
    f = row["last_fail"]
    if f in ("PROCESSOR_TIMEOUT", "AUTH_ERROR"):
        return "1 Retry card today"
    if f == "INSUFFICIENT_FUNDS":
        return "2 Retry the day after payday"
    if f == "DO_NOT_HONOUR":
        return "3 Retry in 2-3 days + bank transfer link"
    if f == "CARD_EXPIRED":
        return "4 Ask to update card / bank transfer link"
    if not row["contactable"]:
        return "5 Email or in-app contact (no SMS)"
    return "6 SMS reminder + bank transfer link"


def load():
    folder = find_data_dir()
    if folder is None:
        return None
    data = {t: pd.read_pickle(os.path.join(folder, f"{t}.pkl")) for t in TABLES}

    # loans
    loans = data["loans"].copy()
    for c in ["term_days", "days_past_due", "total_repayable_gbp", "principal_amount_gbp"]:
        loans[c] = pd.to_numeric(loans[c], errors="coerce")
    for c in ["disbursement_date", "due_date", "closed_date"]:
        loans[c] = pd.to_datetime(loans[c], errors="coerce")
    loans["status"] = loans["loan_status"].str.strip().str.lower()
    closed = loans["closed_date"].notna()
    loans["repaid"] = (closed & (loans["closed_date"] <= loans["due_date"])).astype(int)
    d = loans.drop_duplicates("loan_id").copy()
    asof = d["closed_date"].max()

    # payments
    pay = data["payments"].copy()
    pay["amount_gbp"] = pd.to_numeric(pay["amount_gbp"], errors="coerce")
    pay["payment_status"] = pay["payment_status"].str.strip().str.lower()
    pu = pay.drop_duplicates("payment_id").copy()
    pu["ts"] = pd.to_datetime(pu["payment_datetime_utc"], errors="coerce", utc=True).dt.tz_convert(None)
    pu["failed"] = pu["payment_status"] == "failed"
    pu["month"] = pu["ts"].dt.to_period("M").astype(str)
    paid = (pu[pu["payment_status"] == "success"]
            .groupby("loan_id")["amount_gbp"].sum().rename("paid_gbp"))
    d = d.merge(paid, on="loan_id", how="left")
    d["paid_gbp"] = d["paid_gbp"].fillna(0)

    # customers
    c = data["customers"].drop_duplicates("customer_id").copy()
    c["payday"] = pd.to_numeric(c["payday_day_of_month"], errors="coerce")
    keep = ["customer_id", "region", "device_os", "payday", "employment_status",
            "sms_opt_in", "phone_valid"]
    c = c[[k for k in keep if k in c.columns]]

    # loans already due, with customer fields
    m = d[d["due_date"] <= asof].merge(c, on="customer_id", how="left")
    m["due_month"] = m["due_date"].dt.to_period("M").astype(str)
    m["days_since_payday"] = (m["due_date"].dt.day - m["payday"]) % 30
    m["payday_bucket"] = pd.cut(m["days_since_payday"], [-1, 4, 9, 14, 19, 24, 30],
                                labels=["0-4", "5-9", "10-14", "15-19", "20-24", "25-29"])
    m["contactable"] = (m["sms_opt_in"] == "Y") & (m["phone_valid"] == "Y")

    # recovery worklist
    od = d[d["status"] == "overdue"].merge(c, on="customer_id", how="left")
    od["outstanding_gbp"] = (od["total_repayable_gbp"] - od["paid_gbp"]).clip(lower=0)
    last_fail = pu[pu["failed"]].sort_values("ts").groupby("loan_id")["failure_reason"].last()
    od["last_fail"] = od["loan_id"].map(last_fail)
    od["contactable"] = (od["sms_opt_in"] == "Y") & (od["phone_valid"] == "Y")
    od["action"] = od.apply(_action, axis=1)
    worklist = od.sort_values(["action", "outstanding_gbp"], ascending=[True, False])[
        ["loan_id", "customer_id", "action", "outstanding_gbp",
         "days_past_due", "payday", "last_fail"]].reset_index(drop=True)

    return {"d": d, "m": m, "pu": pu, "worklist": worklist, "asof": asof}


def rate_table(df, col):
    return (df.groupby(col, observed=True)["repaid"]
              .agg(loans="size", on_time="mean").round(3))
