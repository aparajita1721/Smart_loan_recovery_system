#Import libraries
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

#App configuration
st.set_page_config(
    page_title="Smart Loan Recovery System",
    layout="wide"
)

st.title("Smart Loan Recovery System")
st.write("Risk segmentation, recovery prioritization, and borrower-level strategy recommendation.")

#Load data
@st.cache_data
def load_data():
    url = "https://amanxai.com/wp-content/uploads/2025/02/loan-recovery.csv"
    df = pd.read_csv(url)
    return df

df = load_data()

st.sidebar.header("Navigation")
page = st.sidebar.radio(
    "Go to:",
    ["Portfolio Dashboard", "Risk Engine", "Borrower Strategy Assistant"]
)

# Common feature list
FEATURE_COLS = [
    'Age', 'Monthly_Income', 'Loan_Amount', 'Loan_Tenure', 'Interest_Rate',
    'Collateral_Value', 'Outstanding_Loan_Amount', 'Monthly_EMI',
    'Num_Missed_Payments', 'Days_Past_Due'
]

#Train models
#KMeans for segmentation
#Random Forest for High-Risk Prediction
@st.cache_resource
def train_models(data):
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(data[FEATURE_COLS])

    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    borrower_segment = kmeans.fit_predict(df_scaled)
# Map cluster id -> human readable segment
    segment_name_map = {
        0: 'Moderate Income, High Loan Burden',
        1: 'High Income, Low Default Risk',
        2: 'Moderate Income, Medium Risk',
        3: 'High Loan, Higher Default Risk'
    }

    df_model = df.copy()
    df_model['Borrower_Segment'] = borrower_segment
    df_model['Segment_Name'] = df_model['Borrower_Segment'].map(segment_name_map)

    #High-Risk Flag
    df_model['High_Risk_Flag'] = df_model['Segment_Name'].apply(
        lambda x: 1 if x in ['High Loan, Higher Default Risk',
                             'Moderate Income, High Loan Burden'] else 0
    )

    # RF model for probability of being high-risk
    X = df_model[FEATURE_COLS]
    y = df_model['High_Risk_Flag']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    return scaler, kmeans, segment_name_map, rf_model, X_test, df_model

scaler, kmeans, segment_name_map, rf_model, X_test, df_model_full = train_models(df)

# helper: assign recovery strategy
def assign_recovery_strategy(risk_score):
    if risk_score > 0.75:
        return "Immediate legal notices & aggressive recovery attempts"
    elif 0.50 <= risk_score <= 0.75:
        return "Settlement offers & personalized repayment plan"
    else:
        return "Automated reminders & routine monitoring"

#Portfolio Dashboard Page
if page == "Portfolio Dashboard":
    st.header("Portfolio Health Overview")

    col1, col2, col3, col4 = st.columns(4)
    total_loans = len(df)
    recovered_pct = (df["Recovery_Status"].eq("Recovered").mean() * 100.0)
    avg_dpd = df["Days_Past_Due"].mean()
    avg_missed = df["Num_Missed_Payments"].mean()

    col1.metric("Total Loans", f"{total_loans}")
    col2.metric("Recovery %", f"{recovered_pct:.1f}%")
    col3.metric("Avg Days Past Due", f"{avg_dpd:.1f}")
    col4.metric("Avg Missed Payments", f"{avg_missed:.2f}")

    st.markdown("---")

    st.subheader("Loan Amount Distribution & Income Relationship")
    fig1 = px.histogram(
        df,
        x='Loan_Amount',
        nbins=30,
        marginal="violin",
        opacity=0.7,
        title="Loan Amount Distribution",
        labels={'Loan_Amount': "Loan Amount ($)"}
    )

    # density curve
    density_y = px.histogram(df, x='Loan_Amount', nbins=30, histnorm='probability density').data[0]['y']
    fig1.add_trace(go.Scatter(
        x=sorted(df['Loan_Amount']),
        y=density_y,
        mode='lines',
        name='Density Curve',
        line=dict(color='red', width=2)
    ))
    st.plotly_chart(fig1, use_container_width=True)

    st.subheader("Monthly Income vs Loan Amount (Colored by Recovery)")
    fig2 = px.scatter(
        df,
        x='Monthly_Income',
        y='Loan_Amount',
        color='Recovery_Status',
        size='Loan_Amount',
        hover_data={
            'Monthly_Income': True,
            'Loan_Amount': True,
            'Recovery_Status': True
        },
        title="Income vs Loan Amount and Recovery",
        labels={
            "Monthly_Income": "Monthly Income ($)",
            "Loan_Amount": "Loan Amount ($)"
        },
        color_discrete_map={"Recovered": "green", "Not Recovered": "red"}
    )
    fig2.add_annotation(
        x=df['Monthly_Income'].max(),
        y=df['Loan_Amount'].max(),
        text="High income can still recover high loans",
        showarrow=True,
        arrowhead=2,
        font=dict(size=12, color="red")
    )
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Payment History vs Recovery Status")
    fig3 = px.histogram(
        df,
        x="Payment_History",
        color="Recovery_Status",
        barmode="group",
        title="Payment History Impact",
        labels={"Payment_History": "Payment History", "count": "Number of Loans"},
        color_discrete_map={"Recovered": "green", "Not Recovered": "red"}
    )
    st.plotly_chart(fig3, use_container_width=True)

    st.subheader("Missed Payments by Recovery Status")
    fig4 = px.box(
        df,
        x="Recovery_Status",
        y="Num_Missed_Payments",
        title="Missed Payments vs Recovery",
        labels={
            "Recovery_Status": "Recovery Status",
            "Num_Missed_Payments": "Number of Missed Payments"
        },
        color="Recovery_Status",
        color_discrete_map={"Recovered": "green", "Not Recovered": "red"},
        points="all"
    )
    st.plotly_chart(fig4, use_container_width=True)

    st.subheader("Borrower Segments (K-Means)")
    # Attach cluster labels for plotting
    df_segments = df_model_full.copy()
    fig5 = px.scatter(
        df_segments,
        x='Monthly_Income',
        y='Loan_Amount',
        color=df_segments['Borrower_Segment'].astype(str),
        size='Loan_Amount',
        hover_data={
            'Monthly_Income': True,
            'Loan_Amount': True,
            'Borrower_Segment': True,
            'Segment_Name': True
        },
        title="Borrower Segments by Income and Loan Amount",
        labels={
            "Monthly_Income": "Monthly Income ($)",
            "Loan_Amount": "Loan Amount ($)",
            "color": "Segment"
        },
        color_discrete_sequence=px.colors.qualitative.Vivid
    )
    fig5.add_annotation(
        x=df_segments['Monthly_Income'].mean(),
        y=df_segments['Loan_Amount'].max(),
        text="High loans cluster in specific income bands",
        showarrow=True,
        arrowhead=2,
        font=dict(size=12, color="red")
    )
    st.plotly_chart(fig5, use_container_width=True)

    st.markdown("#### Segment Definitions")
    st.write("""
    - **High Income, Low Default Risk**: financially strong, likely to cooperate  
    - **Moderate Income, Medium Risk**: may need flexible plans  
    - **Moderate Income, High Loan Burden**: stretched capacity  
    - **High Loan, Higher Default Risk**: urgent escalation candidates  
    """)

    #Risk Engine
if page == "Risk Engine":
    st.header("Portfolio Risk Scoring & Strategy Recommendation")

    # Predict risk_scores for the held-out test set (X_test)
    preds_proba = rf_model.predict_proba(X_test)[:, 1]

    result_df = X_test.copy()
    result_df['Risk_Score'] = preds_proba
    result_df['Predicted_High_Risk'] = (result_df['Risk_Score'] > 0.5).astype(int)

    # bring back borrower info like Segment_Name, Recovery_Status etc.
    merged = result_df.merge(
        df_model_full[[
            'Borrower_ID', 'Segment_Name', 'Recovery_Status',
            'Collection_Method', 'Collection_Attempts', 'Legal_Action_Taken'
        ]],
        left_index=True,
        right_index=True,
        how='left'
    )

    merged['Recommended_Strategy'] = merged['Risk_Score'].apply(assign_recovery_strategy)

    st.write("Highest-risk accounts first:")

    # sort by risk score desc
    merged_sorted = merged.sort_values(by="Risk_Score", ascending=False)

    st.dataframe(
        merged_sorted[[
            'Borrower_ID',
            'Monthly_Income',
            'Loan_Amount',
            'Num_Missed_Payments',
            'Days_Past_Due',
            'Risk_Score',
            'Predicted_High_Risk',
            'Segment_Name',
            'Recommended_Strategy',
            'Recovery_Status',
            'Collection_Method',
            'Collection_Attempts',
            'Legal_Action_Taken'
        ]].reset_index(drop=True)
    )

    st.download_button(
        label="Download risk report (CSV)",
        data=merged_sorted.to_csv(index=False).encode('utf-8'),
        file_name="loan_recovery_risk_report.csv",
        mime="text/csv"
    )

    #Borrower Strategy Assistant
if page == "Borrower Strategy Assistant":
    st.header("Get Strategy for a Single Borrower")
    st.write("Fill in borrower details and get: segment, risk score, and next action plan.")

    with st.form("borrower_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            age = st.number_input("Age", min_value=18, max_value=100, value=35)
            monthly_income = st.number_input("Monthly Income ($)", min_value=0.0, value=5000.0, step=100.0)
            loan_amount = st.number_input("Loan Amount ($)", min_value=0.0, value=20000.0, step=500.0)
            loan_tenure = st.number_input("Loan Tenure (months)", min_value=1, value=36, step=1)
        with c2:
            interest_rate = st.number_input("Interest Rate (%)", min_value=0.0, value=12.0, step=0.1)
            collateral_value = st.number_input("Collateral Value ($)", min_value=0.0, value=15000.0, step=500.0)
            outstanding_amount = st.number_input("Outstanding Loan Amount ($)", min_value=0.0, value=12000.0, step=500.0)
            monthly_emi = st.number_input("Monthly EMI ($)", min_value=0.0, value=600.0, step=50.0)
        with c3:
            missed_payments = st.number_input("Num Missed Payments", min_value=0, value=2, step=1)
            dpd = st.number_input("Days Past Due", min_value=0, value=15, step=1)

        submitted = st.form_submit_button("🔍 Analyze Borrower")

    if submitted:
        # Build a one-row dataframe from the inputs
        borrower_input = pd.DataFrame([{
            'Age': age,
            'Monthly_Income': monthly_income,
            'Loan_Amount': loan_amount,
            'Loan_Tenure': loan_tenure,
            'Interest_Rate': interest_rate,
            'Collateral_Value': collateral_value,
            'Outstanding_Loan_Amount': outstanding_amount,
            'Monthly_EMI': monthly_emi,
            'Num_Missed_Payments': missed_payments,
            'Days_Past_Due': dpd
        }])

        # ---- Predict Segment via KMeans ----
        scaled_row = scaler.transform(borrower_input[FEATURE_COLS])
        cluster_id = kmeans.predict(scaled_row)[0]
        segment_name = segment_name_map.get(cluster_id, "Unknown Segment")

        # ---- Predict Risk via RF ----
        risk_prob = rf_model.predict_proba(borrower_input[FEATURE_COLS])[:, 1][0]
        strategy = assign_recovery_strategy(risk_prob)

        st.subheader("Borrower Assessment")
        colA, colB, colC = st.columns(3)
        colA.metric("Segment", segment_name)
        colB.metric("Risk Score (0-1)", f"{risk_prob:.2f}")
        colC.metric("Suggested Action", strategy)

        st.markdown("### Action Playbook")
        if "legal" in strategy.lower():
            st.write(
                "- Send immediate legal notice\n"
                "- Freeze further disbursement / credit lines\n"
                "- Prioritize manual follow-up within 24 hrs"
            )
        elif "settlement" in strategy.lower():
            st.write(
                "- Offer restructured EMI / settlement discount\n"
                "- Human agent call to negotiate timeline\n"
                "- Flag for weekly monitoring"
            )
        else:
            st.write(
                "- Send automated reminder SMS / email\n"
                "- Offer self-serve repayment portal link\n"
                "- Review status in next monthly cycle"
            )

        st.success("Analysis complete ✔")

#Footer
st.markdown("---")
st.caption("Smart Loan Recovery System • Prototype dashboard")