import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
from datetime import datetime

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="BankCo Churn Risk Dashboard", layout="wide")

# -----------------------------
# LOAD MODEL + TRAIN COLS
# -----------------------------
pipeline = joblib.load("churn_model_pipeline.joblib")
train_columns = joblib.load("train_columns.joblib")

# -----------------------------
# HISTORY STORAGE
# -----------------------------
HISTORY_FILE = Path("customer_history.csv")

def load_history():
    if HISTORY_FILE.exists():
        return pd.read_csv(HISTORY_FILE)
    return pd.DataFrame()

def save_history_row(row_dict):
    hist = load_history()
    hist = pd.concat([hist, pd.DataFrame([row_dict])], ignore_index=True)
    hist.to_csv(HISTORY_FILE, index=False)

def prepare_features_like_training(raw_df: pd.DataFrame, train_columns: list) -> pd.DataFrame:
    X = pd.get_dummies(raw_df, columns=["Geography", "Gender"], dtype=int)
    for col in train_columns:
        if col not in X.columns:
            X[col] = 0
    X = X[train_columns]
    return X

def risk_band(p):
    # You can adjust these bands
    if p < 0.30:
        return "Low", "✅"
    if p < 0.60:
        return "Medium", "⚠️"
    return "High", "🚨"

# -----------------------------
# HEADER
# -----------------------------
st.title("BankCo — Customer Churn Risk Dashboard")
st.caption(
    "This tool estimates the probability that a customer will **leave the bank** (churn). "
    "Use it to prioritize retention actions. It supports decisions; it does not replace human judgment."
)

with st.expander("How to use this tool (read this first)", expanded=True):
    st.markdown(
        """
**What does the prediction mean?**
- **Exited = 1** → the model believes the customer is **likely to churn** (leave the bank).
- **Exited = 0** → the customer is **less likely to churn**.

**How should you use it?**
- Use the **probability score** to decide whether to contact the customer, offer support, or provide incentives.
- The **threshold** controls when a customer is flagged as “High risk”.
  - Lower threshold → catch more churners (higher recall) but more false alarms.
  - Higher threshold → fewer false alarms but may miss some churners.

**Important:** The model is not perfect. Always combine the score with business context.
        """
    )

# -----------------------------
# SIDEBAR INPUTS
# -----------------------------
@st.cache_data
def load_data_for_ranges():
    df = pd.read_csv("Churn_Modelling.csv")
    # same cleaning as training (just for ranges)
    df = df.drop(["RowNumber", "CustomerId", "Surname"], axis=1, errors="ignore")
    return df

df_ranges = load_data_for_ranges()

def col_minmax(col):
    return float(df_ranges[col].min()), float(df_ranges[col].max())

cs_min, cs_max = col_minmax("CreditScore")
age_min, age_max = col_minmax("Age")
ten_min, ten_max = col_minmax("Tenure")
bal_min, bal_max = col_minmax("Balance")
sal_min, sal_max = col_minmax("EstimatedSalary")

# NumOfProducts can be treated as integer range
prod_min = int(df_ranges["NumOfProducts"].min())
prod_max = int(df_ranges["NumOfProducts"].max())

# -----------------------------
# SIDEBAR INPUTS (DYNAMIC RANGES)
# -----------------------------
st.sidebar.header("Customer Identification")
customer_id = st.sidebar.text_input("Customer ID", value="15634602")
surname = st.sidebar.text_input("Surname", value="Hargrave")

st.sidebar.divider()
st.sidebar.header("Customer Profile (Inputs)")

credit_score = st.sidebar.slider(
    "Credit Score",
    min_value=int(cs_min),
    max_value=int(cs_max),
    value=int(min(max(650, cs_min), cs_max))
)

geography = st.sidebar.selectbox("Geography", ["France", "Germany", "Spain"])
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])

age = st.sidebar.slider(
    "Age",
    min_value=int(age_min),
    max_value=int(age_max),
    value=int(min(max(35, age_min), age_max))
)

tenure = st.sidebar.slider(
    "Tenure (years with bank)",
    min_value=int(ten_min),
    max_value=int(ten_max),
    value=int(min(max(5, ten_min), ten_max))
)

balance = st.sidebar.number_input(
    "Account Balance",
    min_value=float(bal_min),
    max_value=float(bal_max),
    value=float(min(max(50000.0, bal_min), bal_max)),
    step=1000.0
)

num_products = st.sidebar.selectbox(
    "Number of Products",
    list(range(prod_min, prod_max + 1))
)

has_cr_card = st.sidebar.selectbox("Has Credit Card?", ["No", "Yes"])
is_active_member = st.sidebar.selectbox("Is Active Member?", ["No", "Yes"])

estimated_salary = st.sidebar.number_input(
    "Estimated Salary",
    min_value=float(sal_min),
    max_value=float(sal_max),
    value=float(min(max(70000.0, sal_min), sal_max)),
    step=1000.0
)

st.sidebar.divider()
st.sidebar.header("Decision Settings")
threshold = st.sidebar.slider("Risk alert threshold", 0.10, 0.90, 0.50, 0.01)

has_cr_card_val = 1 if has_cr_card == "Yes" else 0
is_active_val = 1 if is_active_member == "Yes" else 0


# -----------------------------
# MAIN LAYOUT
# -----------------------------
left, right = st.columns([1.1, 1.9])

with left:
    st.subheader("Customer Snapshot")

    st.write(f"**Customer ID:** {customer_id}")
    st.write(f"**Surname:** {surname}")
    st.write(f"**Location:** {geography}")
    st.write(f"**Age:** {age}")
    st.write(f"**Active member:** {is_active_member}")
    st.write(f"**Products:** {num_products}")

    st.divider()
    st.subheader("Run Prediction")

    run = st.button("Run prediction", type="primary")

with right:
    st.subheader("Risk Output")

    if not run:
        st.info("Enter customer details on the left, then click **Run prediction**.")
    else:
        # Prepare raw model df (internal)
        model_raw = pd.DataFrame([{
            "CreditScore": credit_score,
            "Geography": geography,
            "Gender": gender,
            "Age": age,
            "Tenure": tenure,
            "Balance": balance,
            "NumOfProducts": num_products,
            "HasCrCard": has_cr_card_val,
            "IsActiveMember": is_active_val,
            "EstimatedSalary": estimated_salary
        }])

        X_input = prepare_features_like_training(model_raw, train_columns)

        proba = float(pipeline.predict_proba(X_input)[0, 1])
        label = 1 if proba >= threshold else 0
        band, icon = risk_band(proba)

        # Risk “gauge” + key metrics
        topA, topB, topC = st.columns([1, 1, 1])
        with topA:
            st.metric("Churn probability", f"{proba:.2%}")
        with topB:
            st.metric("Risk band", f"{icon} {band}")
        with topC:
            st.metric("Prediction", f"Exited = {label}")

        st.progress(min(max(proba, 0.0), 1.0))

        # Recommendations block
        st.markdown("### What to do with this result")
        if label == 1:
            st.warning(
                "**Exited = 1 (likely churn):** Prioritize retention action.\n\n"
                "- Call the customer to understand issues\n"
                "- Offer targeted incentives (fees, rates, products)\n"
                "- Improve engagement (encourage active usage)\n"
                "- Check product fit (NumOfProducts) and service complaints"
            )
        else:
            st.success(
                "**Exited = 0 (lower churn risk):** No urgent retention action.\n\n"
                "- Keep normal engagement\n"
                "- Monitor if risk increases over time (balance/activity changes)"
            )

        # Save to history
        record = {
            "Timestamp": datetime.now().isoformat(timespec="seconds"),
            "CustomerId": str(customer_id),
            "Surname": surname,

            "Geography": geography,
            "Gender": gender,
            "Age": age,
            "Tenure": tenure,
            "Balance": balance,
            "NumOfProducts": num_products,
            "HasCrCard": has_cr_card_val,
            "IsActiveMember": is_active_val,
            "EstimatedSalary": estimated_salary,

            "Churn_Probability": proba,
            "Risk_Band": band,
            "Prediction_Exited": label,
            "Threshold": threshold
        }
        save_history_row(record)
        st.toast("Saved to history.", icon="💾")

# -----------------------------
# HISTORY & TRENDS
# -----------------------------
st.divider()
st.subheader("Customer History & Trends")

hist = load_history()

if hist.empty:
    st.info("No history yet. Run a prediction to start tracking customers.")
else:
    hist["CustomerId"] = hist["CustomerId"].astype(str)
    hist["Timestamp"] = pd.to_datetime(hist["Timestamp"], errors="coerce")

    hleft, hright = st.columns([1, 2])

    with hleft:
        st.markdown("### Select customer")
        customers = sorted(hist["CustomerId"].dropna().unique().tolist())
        selected_customer = st.selectbox("Customer ID", customers)

        # Download history
        csv_bytes = hist.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download all history (CSV)",
            data=csv_bytes,
            file_name="customer_history.csv",
            mime="text/csv"
        )

        st.markdown("### Trend options")
        show_feature_trends = st.checkbox("Show feature trends (Balance, Salary, Activity)", value=True)

    cust = hist[hist["CustomerId"] == str(selected_customer)].copy()
    cust = cust.sort_values("Timestamp")

    with hright:
        st.markdown("### Risk trend (Churn probability over time)")
        if cust["Churn_Probability"].notna().any():
            risk_df = cust[["Timestamp", "Churn_Probability"]].dropna().set_index("Timestamp")
            st.line_chart(risk_df)

        if show_feature_trends:
            st.markdown("### Customer feature trends")
            trend_cols = []
            for c in ["Balance", "EstimatedSalary", "IsActiveMember", "NumOfProducts"]:
                if c in cust.columns:
                    trend_cols.append(c)

            if trend_cols:
                feat_df = cust[["Timestamp"] + trend_cols].dropna().set_index("Timestamp")
                st.line_chart(feat_df)

        st.markdown("### Latest records")
        cols_to_show = [
            "Timestamp", "Surname", "Geography", "Age", "Balance",
            "NumOfProducts", "IsActiveMember",
            "Churn_Probability", "Risk_Band", "Prediction_Exited"
        ]
        cols_to_show = [c for c in cols_to_show if c in cust.columns]
        st.dataframe(cust[cols_to_show].tail(12), use_container_width=True)
