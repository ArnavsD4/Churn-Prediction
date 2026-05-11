import streamlit as st
import pickle
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(page_title="Churn Predictor", page_icon="📊", layout="wide")

st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background: #0d1117;
}
[data-testid="stHeader"] {
    background: transparent;
}
section[data-testid="stSidebar"] {
    background: #161b22;
    border-right: 1px solid #30363d;
}
.main-header {
    text-align: center;
    padding: 2rem 0 1rem;
    border-bottom: 1px solid #21262d;
    margin-bottom: 2rem;
}
.main-header h1 {
    font-size: 2.2rem;
    font-weight: 700;
    color: #e6edf3;
    letter-spacing: -0.5px;
}
.main-header p {
    color: #8b949e;
    font-size: 1rem;
    margin-top: 6px;
}
.section-card {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1.2rem;
}
.section-title {
    font-size: 0.85rem;
    font-weight: 600;
    color: #8b949e;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin-bottom: 1.2rem;
    display: flex;
    align-items: center;
    gap: 8px;
}
.metric-row {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
    margin-bottom: 1.2rem;
}
.metric-box {
    background: #0d1117;
    border: 1px solid #21262d;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    text-align: center;
}
.metric-box .m-label {
    font-size: 0.75rem;
    color: #8b949e;
    margin-bottom: 6px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
.metric-box .m-value {
    font-size: 1.6rem;
    font-weight: 600;
    color: #e6edf3;
}
.verdict-box {
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    display: flex;
    align-items: center;
    gap: 16px;
    margin-bottom: 1.2rem;
}
.verdict-churn {
    background: #1a0a0a;
    border: 1px solid #da3633;
}
.verdict-stay {
    background: #0a1a0f;
    border: 1px solid #238636;
}
.verdict-icon {
    font-size: 2.2rem;
}
.verdict-title {
    font-size: 1.1rem;
    font-weight: 600;
}
.verdict-churn .verdict-title { color: #ff7b72; }
.verdict-stay .verdict-title { color: #56d364; }
.verdict-sub {
    font-size: 0.85rem;
    margin-top: 4px;
}
.verdict-churn .verdict-sub { color: #ffa198; }
.verdict-stay .verdict-sub { color: #7ee787; }
.risk-badge {
    margin-left: auto;
    padding: 4px 14px;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 600;
}
.badge-high { background: #da3633; color: #fff; }
.badge-med  { background: #d29922; color: #fff; }
.badge-low  { background: #238636; color: #fff; }
.rec-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 12px;
}
.rec-card {
    background: #0d1117;
    border: 1px solid #21262d;
    border-radius: 10px;
    padding: 1rem;
}
.rec-card-title {
    font-size: 0.88rem;
    font-weight: 600;
    color: #e6edf3;
    margin-bottom: 6px;
}
.rec-card-text {
    font-size: 0.8rem;
    color: #8b949e;
    line-height: 1.5;
}
.predict-btn > button {
    background: #1f6feb !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
    padding: 0.6rem 2rem !important;
    width: 100% !important;
    transition: 0.2s !important;
}
.predict-btn > button:hover {
    background: #388bfd !important;
}
.footer-text {
    text-align: center;
    color: #484f58;
    font-size: 0.78rem;
    padding: 1.5rem 0;
    border-top: 1px solid #21262d;
    margin-top: 1rem;
}
div[data-testid="stSlider"] > div {
    padding-top: 0.3rem;
}
</style>
""", unsafe_allow_html=True)

model = pickle.load(open('model.pkl', 'rb'))

st.markdown("""
<div class="main-header">
    <h1>📊 Customer Churn Predictor</h1>
    <p>AI-powered churn risk analysis with retention insights — powered by XGBoost</p>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="section-card"><div class="section-title">📋 Key customer details</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
with col1:
    tenure = st.slider('Tenure (months)', 0, 72, 12)
    contract = st.selectbox('Contract type', [0, 1, 2],
                            format_func=lambda x: ['Month-to-month', 'One year', 'Two year'][x])
with col2:
    monthly = st.number_input('Monthly charges ($)', 0.0, 150.0, 65.0, step=1.0)
    internet = st.selectbox('Internet service', [0, 1, 2],
                            format_func=lambda x: ['DSL', 'Fiber optic', 'No internet'][x])
with col3:
    total = st.number_input('Total charges ($)', 0.0, 10000.0, 780.0, step=10.0)
    payment = st.selectbox('Payment method', [0, 1, 2, 3],
                           format_func=lambda x: ['Electronic check', 'Mailed check',
                                                   'Bank transfer', 'Credit card'][x])

st.markdown('</div>', unsafe_allow_html=True)

with st.expander("⚙️  Advanced options — pre-filled with common defaults"):
    c1, c2, c3 = st.columns(3)
    with c1:
        gender     = st.selectbox('Gender', [0, 1], format_func=lambda x: 'Female' if x==0 else 'Male')
        senior     = st.selectbox('Senior citizen', [0, 1], format_func=lambda x: 'No' if x==0 else 'Yes')
        partner    = st.selectbox('Has partner', [0, 1], format_func=lambda x: 'No' if x==0 else 'Yes')
        dependents = st.selectbox('Has dependents', [0, 1], format_func=lambda x: 'No' if x==0 else 'Yes')
    with c2:
        phone      = st.selectbox('Phone service', [1, 0], format_func=lambda x: 'Yes' if x==1 else 'No')
        multiple   = st.selectbox('Multiple lines', [0, 1, 2], format_func=lambda x: ['No', 'Yes', 'No phone'][x])
        security   = st.selectbox('Online security', [0, 1, 2], format_func=lambda x: ['No', 'Yes', 'No internet'][x])
        backup     = st.selectbox('Online backup', [0, 1, 2], format_func=lambda x: ['No', 'Yes', 'No internet'][x])
    with c3:
        protection = st.selectbox('Device protection', [0, 1, 2], format_func=lambda x: ['No', 'Yes', 'No internet'][x])
        support    = st.selectbox('Tech support', [0, 1, 2], format_func=lambda x: ['No', 'Yes', 'No internet'][x])
        tv         = st.selectbox('Streaming TV', [0, 1, 2], format_func=lambda x: ['No', 'Yes', 'No internet'][x])
        movies     = st.selectbox('Streaming movies', [0, 1, 2], format_func=lambda x: ['No', 'Yes', 'No internet'][x])
        paperless  = st.selectbox('Paperless billing', [1, 0], format_func=lambda x: 'Yes' if x==1 else 'No')

st.markdown("<br>", unsafe_allow_html=True)

col_btn = st.columns([1, 2, 1])
with col_btn[1]:
    st.markdown('<div class="predict-btn">', unsafe_allow_html=True)
    predict = st.button('🔍  Predict churn risk')
    st.markdown('</div>', unsafe_allow_html=True)

if predict:
    input_data = pd.DataFrame(
        [[gender, senior, partner, dependents, tenure,
          phone, multiple, internet, security, backup,
          protection, support, tv, movies, contract,
          paperless, payment, monthly, total]],
        columns=['gender','SeniorCitizen','Partner','Dependents','tenure',
                 'PhoneService','MultipleLines','InternetService','OnlineSecurity',
                 'OnlineBackup','DeviceProtection','TechSupport','StreamingTV',
                 'StreamingMovies','Contract','PaperlessBilling','PaymentMethod',
                 'MonthlyCharges','TotalCharges'])

    prediction  = model.predict(input_data)[0]
    prob        = round(model.predict_proba(input_data)[0][1] * 100, 1)
    will_churn  = prediction == 1

    loyalty = round(max(5, min(98, 100 - prob * 0.6 + tenure / 72 * 20)))
    services = sum([phone==1, multiple==1, security==1, backup==1,
                    protection==1, support==1, tv==1, movies==1])
    risk_label = "High" if prob >= 60 else "Medium" if prob >= 30 else "Low"
    badge_cls  = "badge-high" if prob >= 60 else "badge-med" if prob >= 30 else "badge-low"

    st.markdown("---")

    # ── VERDICT ──────────────────────────────────────────────────────────────
    if will_churn:
        st.markdown(f"""
        <div class="verdict-box verdict-churn">
            <div class="verdict-icon">⚠️</div>
            <div>
                <div class="verdict-title">This customer is likely to churn</div>
                <div class="verdict-sub">Immediate retention action recommended — offer a better plan or discount.</div>
            </div>
            <span class="risk-badge {badge_cls}">{risk_label} risk</span>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="verdict-box verdict-stay">
            <div class="verdict-icon">✅</div>
            <div>
                <div class="verdict-title">This customer is likely to stay</div>
                <div class="verdict-sub">Customer appears satisfied. Continue monitoring and reward loyalty.</div>
            </div>
            <span class="risk-badge {badge_cls}">{risk_label} risk</span>
        </div>""", unsafe_allow_html=True)

    # ── METRICS ───────────────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="metric-row">
        <div class="metric-box"><div class="m-label">Churn probability</div><div class="m-value">{prob}%</div></div>
        <div class="metric-box"><div class="m-label">Loyalty score</div><div class="m-value">{loyalty}/100</div></div>
        <div class="metric-box"><div class="m-label">Active services</div><div class="m-value">{services}/8</div></div>
        <div class="metric-box"><div class="m-label">Risk level</div><div class="m-value">{risk_label}</div></div>
    </div>""", unsafe_allow_html=True)

    # ── CHARTS ────────────────────────────────────────────────────────────────
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        gauge_color = '#da3633' if prob >= 60 else '#d29922' if prob >= 30 else '#238636'
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob,
            domain={'x':[0,1],'y':[0,1]},
            title={'text':"Churn risk score", 'font':{'color':'#8b949e','size':14}},
            number={'suffix':"%",'font':{'color':'#e6edf3','size':32}},
            gauge={
                'axis':{'range':[0,100],'tickcolor':'#484f58','tickfont':{'color':'#8b949e'}},
                'bar':{'color':gauge_color},
                'bgcolor':'#161b22',
                'bordercolor':'#21262d',
                'steps':[
                    {'range':[0,30],'color':'#0a1a0f'},
                    {'range':[30,60],'color':'#1a1500'},
                    {'range':[60,100],'color':'#1a0a0a'}
                ],
                'threshold':{
                    'line':{'color':'#e6edf3','width':2},
                    'thickness':0.75,
                    'value':prob
                }
            }
        ))
        fig_gauge.update_layout(
            paper_bgcolor='#161b22',
            plot_bgcolor='#161b22',
            font={'color':'#e6edf3'},
            height=280,
            margin=dict(l=20,r=20,t=50,b=10)
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

    with chart_col2:
        feature_names = ['gender','SeniorCitizen','Partner','Dependents','tenure',
                         'PhoneService','MultipleLines','InternetService','OnlineSecurity',
                         'OnlineBackup','DeviceProtection','TechSupport','StreamingTV',
                         'StreamingMovies','Contract','PaperlessBilling','PaymentMethod',
                         'MonthlyCharges','TotalCharges']
        feat_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=True).tail(7)

        colors = ['#da3633' if v > 0.12 else '#d29922' if v > 0.07 else '#1f6feb'
                  for v in feat_df['Importance']]

        fig_bar = go.Figure(go.Bar(
            x=feat_df['Importance'],
            y=feat_df['Feature'],
            orientation='h',
            marker_color=colors,
            marker_line_width=0
        ))
        fig_bar.update_layout(
            title=dict(text='Top churn factors', font=dict(color='#8b949e', size=14)),
            paper_bgcolor='#161b22',
            plot_bgcolor='#161b22',
            font={'color':'#8b949e'},
            height=280,
            margin=dict(l=10,r=20,t=50,b=10),
            xaxis=dict(gridcolor='#21262d',color='#8b949e',showgrid=True),
            yaxis=dict(gridcolor='#21262d',color='#e6edf3',showgrid=False)
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # ── RECOMMENDATIONS ───────────────────────────────────────────────────────
    if contract == 0:
        r1 = ("📄 Upgrade contract",
              "Offer a discount to switch from month-to-month to a 1 or 2 year plan — the #1 churn reducer.")
    else:
        r1 = ("📄 Contract is stable ✅",
              "Long-term contract significantly reduces churn risk. No action needed here.")

    if monthly > 80:
        r2 = ("💰 Offer a discount",
              "Monthly charges are high. A 10–15% loyalty discount could retain this customer.")
    else:
        r2 = ("💰 Pricing is fair ✅",
              "Monthly charges are in a comfortable range. No pricing action needed right now.")

    if tenure < 12:
        r3 = ("🎁 New customer care",
              "Customer is relatively new. Send a welcome loyalty offer and assign priority support.")
    else:
        r3 = ("🏆 Reward loyalty ✅",
              "Long-tenure customer. A loyalty bonus or early renewal discount will strengthen retention.")

    st.markdown(f"""
    <div class="section-card">
        <div class="section-title">💡 Recommended actions</div>
        <div class="rec-grid">
            <div class="rec-card"><div class="rec-card-title">{r1[0]}</div><div class="rec-card-text">{r1[1]}</div></div>
            <div class="rec-card"><div class="rec-card-title">{r2[0]}</div><div class="rec-card-text">{r2[1]}</div></div>
            <div class="rec-card"><div class="rec-card-title">{r3[0]}</div><div class="rec-card-text">{r3[1]}</div></div>
        </div>
    </div>""", unsafe_allow_html=True)

st.markdown("""
<div class="footer-text">
    Built by Arnav Dasmal &nbsp;|&nbsp; NIT Durgapur &nbsp;|&nbsp; Customer Churn Prediction using XGBoost + Streamlit
</div>""", unsafe_allow_html=True)
