"""Shared data loading, cleaning and roll-to-default scoring for app.py and the notebook.

Looks for loans.pkl, customers.pkl and payments.pkl in ./data/raw, ./Data/raw,
./data, ./Data, ./notebooks or the current folder.
"""
import os

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TABLES = ["loans", "customers", "payments"]
CANDIDATE_DIRS = ["data/raw", "Data/raw", "data", "Data", "notebooks", "."]
RECENT_FROM = "2026-05-01"  # processor change / deterioration starts here

WORK_WINDOW_DPD = 21
SNAPSHOT_DAYS = [1, 3, 5, 7, 10, 14, 21]   # days past due at which training snapshots are taken
RESOLVED_AFTER_DAYS = 91                   # write-off happens after 90 days past due

# Roll-to-default model inputs. Age is deliberately excluded (protected characteristic).
RISK_NUM = ["term_days", "principal_amount_gbp", "interest_ratio", "customer_loan_number",
            "days_since_payday", "contactable", "n_fail_due", "outstanding_at_due",
            "dpd", "post_attempts", "post_fails", "post_paid", "days_since_last_attempt",
            "balance_now"]
RISK_CAT = ["repayment_method", "employment_status", "monthly_income_band",
            "acquisition_channel", "region", "last_fail_reason_due", "last_processor_due"]


def find_data_dir():
    for folder in CANDIDATE_DIRS:
        if all(os.path.exists(os.path.join(folder, f"{t}.pkl")) for t in TABLES):
            return folder
    return None


def _action(row):
    if row["days_past_due"] > WORK_WINDOW_DPD:
        return "7 Recovery track: final notice / write-off review"
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


# --------------------------------------------------------------------------- features
def due_date_features(d, pu):
    """Payment behaviour up to the end of each loan's due date."""
    p = pu.merge(d[["loan_id", "due_date"]], on="loan_id")
    p = p[p["ts"] < p["due_date"] + pd.Timedelta(days=1)]
    last = p.sort_values("ts").groupby("loan_id").last()
    f = pd.DataFrame({
        "n_fail_due": p.groupby("loan_id")["failed"].sum(),
        "paid_by_due": p[p["payment_status"] == "success"].groupby("loan_id")["amount_gbp"].sum(),
        "last_fail_reason_due": last["failure_reason"],
        "last_processor_due": last["processor"],
    })
    out = d.merge(f, left_on="loan_id", right_index=True, how="left")
    out[["n_fail_due", "paid_by_due"]] = out[["n_fail_due", "paid_by_due"]].fillna(0)
    out["outstanding_at_due"] = (out["total_repayable_gbp"] - out["paid_by_due"]).clip(lower=0)
    return out


def post_due_features(loans, pu):
    """Activity between the due date and the snapshot day. `loans` must have a `dpd` column."""
    p = pu.merge(loans[["loan_id", "due_date", "dpd"]], on="loan_id")
    p["day"] = (p["ts"].dt.normalize() - p["due_date"]).dt.days
    p = p[(p["day"] >= 1) & (p["day"] < p["dpd"])]
    g = p.groupby("loan_id")
    out = loans.copy()
    out["post_attempts"] = out["loan_id"].map(g.size()).fillna(0)
    out["post_fails"] = out["loan_id"].map(g["failed"].sum()).fillna(0)
    out["post_paid"] = out["loan_id"].map(
        p[p["payment_status"] == "success"].groupby("loan_id")["amount_gbp"].sum()).fillna(0)
    out["days_since_last_attempt"] = out["dpd"] - out["loan_id"].map(g["day"].max()).fillna(0)
    out["balance_now"] = (out["outstanding_at_due"] - out["post_paid"]).clip(lower=0)
    return out


def build_snapshots(m, pu, asof):
    """One row per resolved late loan per snapshot day it was still unpaid. Target: written_off."""
    late = m[(m["repaid"] == 0) & (m["due_date"] <= asof - pd.Timedelta(days=RESOLVED_AFTER_DAYS))]
    late = late.assign(written_off=(late["status"] == "written off").astype(int))
    closed_day = (late["closed_date"] - late["due_date"]).dt.days
    snaps = [post_due_features(late[~(closed_day < k)].assign(dpd=k), pu) for k in SNAPSHOT_DAYS]
    return pd.concat(snaps, ignore_index=True)


# --------------------------------------------------------------------------- model
def make_risk_model():
    pre = ColumnTransformer([
        ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), RISK_NUM),
        ("cat", make_pipeline(SimpleImputer(strategy="constant", fill_value="NA"),
                              OneHotEncoder(handle_unknown="ignore", min_frequency=20)), RISK_CAT),
    ])
    return make_pipeline(pre, LogisticRegression(max_iter=3000, C=0.5))


def evaluate_risk_model(snaps, test_frac=0.3, capacity=0.2):
    """Time-ordered holdout: train on older loans, test on the newest `test_frac` by due date."""
    cut = snaps.drop_duplicates("loan_id")["due_date"].quantile(1 - test_frac)
    tr, te = snaps[snaps["due_date"] <= cut], snaps[snaps["due_date"] > cut].copy()
    mdl = make_risk_model().fit(tr[RISK_NUM + RISK_CAT], tr["written_off"])
    te["p"] = mdl.predict_proba(te[RISK_NUM + RISK_CAT])[:, 1]

    # day-1 call list: work the top `capacity` share of newly overdue loans
    t1 = te[te["dpd"] == 1].copy()
    t1["exp_loss"] = t1["p"] * t1["balance_now"]
    t1["random"] = np.random.default_rng(0).random(len(t1))
    lost = (t1["balance_now"] * t1["written_off"]).sum()
    k = int(np.ceil(len(t1) * capacity))
    reach = {lab: (t1.nlargest(k, col).eval("balance_now * written_off").sum() / lost)
             for lab, col in [("Alphabetical (random)", "random"), ("Largest balance", "balance_now"),
                              ("Model P(write-off)", "p"), ("Model P x balance", "exp_loss")]}
    return {
        "train_loans": tr["loan_id"].nunique(), "test_loans": te["loan_id"].nunique(),
        "test_from": cut, "auc": roc_auc_score(te["written_off"], te["p"]),
        "auc_dpd_only": roc_auc_score(te["written_off"], te["dpd"]),
        "auc_by_dpd": te.groupby("dpd").apply(
            lambda g: roc_auc_score(g["written_off"], g["p"]), include_groups=False),
        "capacity": capacity, "gbp_reached": pd.Series(reach), "test": te,
    }


# --------------------------------------------------------------------------- load
def load():
    folder = find_data_dir()
    if folder is None:
        return None
    data = {t: pd.read_pickle(os.path.join(folder, f"{t}.pkl")) for t in TABLES}

    # loans
    loans = data["loans"].copy()
    for c in ["term_days", "days_past_due", "total_repayable_gbp", "principal_amount_gbp",
              "customer_loan_number"]:
        loans[c] = pd.to_numeric(loans[c], errors="coerce")
    for c in ["disbursement_date", "due_date", "closed_date"]:
        loans[c] = pd.to_datetime(loans[c], errors="coerce")
    loans["status"] = loans["loan_status"].str.strip().str.lower()
    closed = loans["closed_date"].notna()
    loans["repaid"] = (closed & (loans["closed_date"] <= loans["due_date"])).astype(int)
    d = loans.drop_duplicates("loan_id").copy()
    d["interest_ratio"] = d["total_repayable_gbp"] / d["principal_amount_gbp"] - 1
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
    d = due_date_features(d, pu)

    # customers
    c = data["customers"].drop_duplicates("customer_id").copy()
    c["payday"] = pd.to_numeric(c["payday_day_of_month"], errors="coerce")
    keep = ["customer_id", "region", "device_os", "payday", "employment_status",
            "monthly_income_band", "acquisition_channel", "sms_opt_in", "phone_valid"]
    c = c[[k for k in keep if k in c.columns]]
    d = d.merge(c, on="customer_id", how="left")
    d["days_since_payday"] = (d["due_date"].dt.day - d["payday"]) % 30
    d["contactable"] = ((d["sms_opt_in"] == "Y") & (d["phone_valid"] == "Y")).astype(int)

    # loans already due
    m = d[d["due_date"] <= asof].copy()
    m["due_month"] = m["due_date"].dt.to_period("M").astype(str)
    m["payday_bucket"] = pd.cut(m["days_since_payday"], [-1, 4, 9, 14, 19, 24, 30],
                                labels=["0-4", "5-9", "10-14", "15-19", "20-24", "25-29"])

    # roll-to-default model: evaluate on a time holdout, then refit on all resolved late loans
    snaps = build_snapshots(m, pu, asof)
    model_eval = evaluate_risk_model(snaps)
    risk_model = make_risk_model().fit(snaps[RISK_NUM + RISK_CAT], snaps["written_off"])

    # recovery worklist: score each overdue loan as of today
    od = d[d["status"] == "overdue"].copy()
    od["dpd"] = (asof - od["due_date"]).dt.days.clip(lower=1)
    od = post_due_features(od, pu)
    od["outstanding_gbp"] = (od["total_repayable_gbp"] - od["paid_gbp"]).clip(lower=0)
    od["p_writeoff"] = risk_model.predict_proba(od[RISK_NUM + RISK_CAT])[:, 1]
    od["expected_loss_gbp"] = od["p_writeoff"] * od["outstanding_gbp"]
    last_fail = pu[pu["failed"]].sort_values("ts").groupby("loan_id")["failure_reason"].last()
    od["last_fail"] = od["loan_id"].map(last_fail)
    od["days_past_due"] = od["days_past_due"].fillna(od["dpd"])
    od["stage"] = np.where(od["days_past_due"] <= WORK_WINDOW_DPD,
                           f"Work now (1-{WORK_WINDOW_DPD} dpd)",
                           f"Recovery track (>{WORK_WINDOW_DPD} dpd)")
    od["contactable"] = od["contactable"].astype(bool)
    od["action"] = od.apply(_action, axis=1)
    worklist = (od.sort_values(["stage", "expected_loss_gbp"], ascending=[False, False])
                  .reset_index(drop=True))
    worklist.insert(0, "priority", np.arange(1, len(worklist) + 1))
    worklist = worklist[["priority", "loan_id", "customer_id", "stage", "action", "p_writeoff",
                         "expected_loss_gbp", "outstanding_gbp", "days_past_due", "payday",
                         "last_fail", "contactable"]]

    return {"d": d, "m": m, "pu": pu, "worklist": worklist, "asof": asof,
            "model_eval": model_eval, "risk_model": risk_model, "snaps": snaps}


def predict_one(risk_model, snaps, **inputs):
    """Score one hypothetical overdue loan. Fields not given take typical training values."""
    row = {c: snaps[c].median() for c in RISK_NUM}
    row.update({c: snaps[c].mode().iat[0] for c in RISK_CAT})
    row.update(inputs)
    x = pd.DataFrame([row])[RISK_NUM + RISK_CAT]
    return float(risk_model.predict_proba(x)[:, 1][0])


def rate_table(df, col):
    return (df.groupby(col, observed=True)["repaid"]
              .agg(loans="size", on_time="mean").round(3))
