"""Oriel Collections dashboard.

Run from the project root:  streamlit run app.py
Data loading lives in core.py (shared with api.py).
"""
import altair as alt
import numpy as np
import streamlit as st

import core

st.set_page_config(page_title="Oriel Collections", layout="wide")


@st.cache_data(show_spinner="Loading and cleaning data...")
def load():
    return core.load()


def show(df, fmt=None):
    """Display a dataframe with number formatting."""
    styler = df.style.format(fmt or {}, na_rep="")
    st.dataframe(styler, width="stretch")


PCT = "{:.1%}"
INT = "{:,.0f}"
GBP = "£{:,.0f}"

st.title("Oriel Collections")

res = load()
if res is None:
    st.error("Could not find loans.pkl, customers.pkl and payments.pkl in data/raw, "
             "Data/raw or the project folder. Run the first cell of the notebook "
             "to create them.")
    st.stop()

d, m, pu, worklist, asof = res["d"], res["m"], res["pu"], res["worklist"], res["asof"]
st.caption(f"Data as of {asof.date()} | {len(d):,} unique loans")

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Overview", "Payment failures", "Repayment drivers", "Recovery worklist", "Predict"])

# ---- Overview
with tab1:
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("On-time rate (due loans)", f"{m['repaid'].mean() * 100:.1f}%")
    k2.metric("Collected / due",
              f"{m['paid_gbp'].sum() / m['total_repayable_gbp'].sum() * 100:.1f}%")
    k3.metric("Overdue loans", f"{len(worklist):,}")
    k4.metric("Outstanding (overdue)", f"£{worklist['outstanding_gbp'].sum():,.0f}")

    st.subheader("On-time rate by due month")
    t = m.groupby("due_month")["repaid"].agg(loans="size", on_time="mean").reset_index()
    chart = (alt.Chart(t).mark_line(point=True)
             .encode(x=alt.X("due_month:N", title="Due month"),
                     y=alt.Y("on_time:Q", title="On-time rate",
                             scale=alt.Scale(domain=[0.7, 1.0]),
                             axis=alt.Axis(format="%")),
                     tooltip=["due_month", "loans",
                              alt.Tooltip("on_time:Q", format=".1%")]))
    st.altair_chart(chart, width="stretch")
    st.info("On-time rate falls from about 91% to about 78-80% for loans due from "
            "May 2026. In the same month, the payment failure rate of PROC_B jumped "
            "from 0% to about 18%.")
    show(t.set_index("due_month"), {"on_time": PCT, "loans": INT})

# ---- Payment failures
with tab2:
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("By processor")
        show(pu.groupby("processor")["failed"].agg(attempts="size", fail_rate="mean")
               .sort_values("fail_rate", ascending=False),
             {"attempts": INT, "fail_rate": PCT})
    with c2:
        st.subheader("By payment method")
        show(pu.groupby("payment_method")["failed"].agg(attempts="size", fail_rate="mean")
               .sort_values("fail_rate", ascending=False),
             {"attempts": INT, "fail_rate": PCT})

    st.subheader("Failure reasons")
    st.bar_chart(pu.loc[pu["failed"], "failure_reason"].value_counts())

    st.subheader("Monthly fail rate by processor")
    st.line_chart(pu.pivot_table(index="month", columns="processor",
                                 values="failed", aggfunc="mean").round(3))

# ---- Repayment drivers
with tab3:
    recent = m[m["due_date"] >= core.RECENT_FROM]
    st.caption(f"Loans due from {core.RECENT_FROM} ({len(recent):,} loans)")
    cols = st.columns(2)
    for i, col in enumerate(["sms_opt_in", "phone_valid", "repayment_method", "payday_bucket"]):
        with cols[i % 2]:
            st.subheader(f"On-time rate by {col}")
            show(core.rate_table(recent, col), {"loans": INT, "on_time": PCT})
    st.subheader("Contactable x repayment method")
    show(core.rate_table(recent, ["contactable", "repayment_method"]),
         {"loans": INT, "on_time": PCT})

# ---- Recovery worklist
with tab4:
    ev = res["model_eval"]
    st.subheader("Daily prioritised call list")
    st.caption(
        f"Ranked by expected loss = P(write-off) x outstanding balance. Loans more than "
        f"{core.WORK_WINDOW_DPD} days past due go to the recovery track: in the history, "
        "no late loan was repaid after about 31 days past due.")

    summary = (worklist.groupby(["stage", "action"])
               .agg(loans=("loan_id", "size"),
                    outstanding_gbp=("outstanding_gbp", "sum"),
                    expected_loss_gbp=("expected_loss_gbp", "sum"),
                    avg_p_writeoff=("p_writeoff", "mean"),
                    avg_days_past_due=("days_past_due", "mean")))
    show(summary, {"loans": INT, "outstanding_gbp": GBP, "expected_loss_gbp": GBP,
                   "avg_p_writeoff": PCT, "avg_days_past_due": INT})

    f1, f2, f3 = st.columns(3)
    all_stages = sorted(worklist["stage"].unique(), reverse=True)
    stages = f1.multiselect("Stage", all_stages, default=all_stages[:1])
    all_actions = sorted(worklist["action"].unique())
    actions = f2.multiselect("Action", all_actions, default=all_actions)
    min_out = f3.number_input("Minimum outstanding (GBP)", min_value=0.0, value=0.0, step=100.0)

    view = worklist[worklist["stage"].isin(stages) & worklist["action"].isin(actions)
                    & (worklist["outstanding_gbp"] >= min_out)]
    st.write(f"{len(view):,} loans | £{view['outstanding_gbp'].sum():,.0f} outstanding | "
             f"£{view['expected_loss_gbp'].sum():,.0f} expected loss")
    st.dataframe(view.style.format({"outstanding_gbp": "£{:,.2f}", "expected_loss_gbp": "£{:,.2f}",
                                    "p_writeoff": PCT, "days_past_due": INT, "payday": INT},
                                   na_rep=""),
                 width="stretch", hide_index=True)

    with st.expander("How good is the roll-to-default model?"):
        st.write(
            f"Logistic regression trained on {ev['train_loans']:,} late loans, tested on the "
            f"{ev['test_loans']:,} newest (due after {ev['test_from'].date()}). "
            f"Test AUC {ev['auc']:.3f} vs {ev['auc_dpd_only']:.3f} using days past due alone. "
            "Ranking power is modest in the first week past due, so use the score to order "
            "the list, not as a verdict on a customer.")
        st.write(f"Share of later write-off £ reached by calling the top "
                 f"{ev['capacity']:.0%} of newly overdue loans:")
        show(ev["gbp_reached"].rename("gbp_reached").to_frame(), {"gbp_reached": PCT})
        show(ev["auc_by_dpd"].rename("AUC").rename_axis("days_past_due").to_frame(),
             {"AUC": "{:.3f}"})
    st.download_button("Download worklist (CSV)", view.to_csv(index=False),
                       file_name="recovery_worklist.csv", mime="text/csv")

# ---- Predict: score a single overdue loan
with tab5:
    st.subheader("Predict roll into default for one overdue loan")
    st.caption("Enter what you know about the loan today. Fields not shown take typical values "
               "from past late loans. The model is the same one that ranks the call list.")
    snaps, risk_model = res["snaps"], res["risk_model"]
    opts = lambda col: sorted(snaps[col].dropna().unique().tolist())

    with st.form("predict"):
        a, b, c = st.columns(3)
        dpd = a.number_input("Days past due", 1, 90, 3)
        balance = a.number_input("Outstanding balance (GBP)", 0.0, 5000.0, 400.0, step=50.0)
        term = a.number_input("Loan term (days)", 7, 120, 30)
        method = b.selectbox("Repayment method", opts("repayment_method"))
        reason = b.selectbox("Last failure reason on/before due date",
                             ["None (no failed attempt)"] + opts("last_fail_reason_due"))
        contactable = b.checkbox("Contactable (SMS opt-in and valid phone)", value=True)
        post_paid = c.number_input("Paid since due date (GBP)", 0.0, 5000.0, 0.0, step=10.0)
        post_fails = c.number_input("Failed attempts since due date", 0, 10, 0)
        since_last = c.number_input("Days since last payment attempt", 0, 90, int(dpd))
        employment = st.selectbox("Employment status", opts("employment_status"))
        submitted = st.form_submit_button("Predict")

    if submitted:
        failed = reason != "None (no failed attempt)"
        p = core.predict_one(
            risk_model, snaps,
            dpd=dpd, balance_now=balance, outstanding_at_due=balance + post_paid,
            term_days=term, repayment_method=method, contactable=int(contactable),
            last_fail_reason_due=reason if failed else np.nan, n_fail_due=int(failed),
            post_paid=post_paid, post_fails=post_fails,
            post_attempts=post_fails + (1 if post_paid > 0 else 0),
            days_since_last_attempt=since_last, employment_status=employment)
        action = core._action({"days_past_due": dpd, "contactable": contactable,
                               "last_fail": reason if failed else None})
        r1, r2, r3 = st.columns(3)
        r1.metric("P(written off)", f"{p:.0%}")
        r2.metric("Expected loss", f"£{p * balance:,.0f}")
        r3.metric("Stage", "Work now" if dpd <= core.WORK_WINDOW_DPD else "Recovery track")
        st.success(f"Recommended action: {action}")
        st.caption(f"Average write-off rate of late loans in training: "
                   f"{snaps.drop_duplicates('loan_id')['written_off'].mean():.0%}. "
                   "Use this as a ranking aid, not a verdict on the customer.")
