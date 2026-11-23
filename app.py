"""Oriel Collections dashboard.

Run from the project root:  streamlit run app.py
Data loading lives in core.py (shared with api.py).
"""
import altair as alt
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

tab1, tab2, tab3, tab4 = st.tabs(
    ["Overview", "Payment failures", "Repayment drivers", "Recovery worklist"])

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
    st.subheader("Overdue loans and recommended action")
    summary = (worklist.groupby("action")
               .agg(loans=("loan_id", "size"),
                    outstanding_gbp=("outstanding_gbp", "sum"),
                    avg_days_past_due=("days_past_due", "mean")))
    show(summary, {"loans": INT, "outstanding_gbp": GBP, "avg_days_past_due": INT})

    f1, f2 = st.columns(2)
    all_actions = sorted(worklist["action"].unique())
    actions = f1.multiselect("Action", all_actions, default=all_actions)
    min_out = f2.number_input("Minimum outstanding (GBP)", min_value=0.0, value=0.0, step=100.0)

    view = worklist[worklist["action"].isin(actions) & (worklist["outstanding_gbp"] >= min_out)]
    st.write(f"{len(view):,} loans | £{view['outstanding_gbp'].sum():,.0f} outstanding")
    st.dataframe(view.style.format({"outstanding_gbp": "£{:,.2f}", "days_past_due": INT,
                                    "payday": INT}, na_rep=""),
                 width="stretch", hide_index=True)
    st.download_button("Download worklist (CSV)", view.to_csv(index=False),
                       file_name="recovery_worklist.csv", mime="text/csv")
