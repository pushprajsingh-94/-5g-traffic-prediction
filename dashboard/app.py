import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(
    page_title="5G Traffic Predictor",
    page_icon="📡",
    layout="wide"
)

st.markdown("""
<style>
    .main { background-color: #0a0a1a; }
    .stApp { background-color: #0a0a1a; }
    h1, h2, h3 { color: #00d4ff !important; }
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border: 1px solid #00d4ff;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        margin: 5px;
    }
    .metric-value { font-size: 2rem; font-weight: bold; color: #00d4ff; }
    .metric-label { font-size: 0.85rem; color: #8892b0; margin-top: 4px; }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_traffic_data():
    return pd.read_csv('data/5g_traffic.csv', parse_dates=['timestamp'])

@st.cache_data
def load_predictions():
    results = {}
    files = {
        'Random Forest': 'results/rf_predictions.csv',
        'ARIMA': 'results/arima_predictions.csv',
        'Linear Regression': 'results/lr_predictions.csv'
    }
    for name, path in files.items():
        if os.path.exists(path):
            results[name] = pd.read_csv(path)
    return results

df = load_traffic_data()
predictions = load_predictions()

st.sidebar.markdown("## 📡 5G Traffic Predictor")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigate", [
    "📊 Data Overview",
    "🔮 Model Predictions",
    "📈 Model Comparison",
    "🚀 24Hr Forecast"
])
st.sidebar.markdown("---")
st.sidebar.markdown("**By:** Pushpraj Singh")
st.sidebar.markdown("**Stack:** Python · Random Forest · ARIMA · Linear Regression")

# PAGE 1
if page == "📊 Data Overview":
    st.title("📡 5G Network Traffic — Data Overview")

    col1, col2, col3, col4 = st.columns(4)
    metrics = [
        (f"{df['traffic_mbps'].mean():.0f}", "Avg Traffic (Mbps)"),
        (f"{df['traffic_mbps'].max():.0f}", "Peak Traffic (Mbps)"),
        (f"{df['traffic_mbps'].min():.0f}", "Min Traffic (Mbps)"),
        (f"{len(df):,}", "Total Records"),
    ]
    for col, (val, label) in zip([col1, col2, col3, col4], metrics):
        with col:
            st.markdown(f"""<div class='metric-card'>
                <div class='metric-value'>{val}</div>
                <div class='metric-label'>{label}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("📈 Traffic Over Time")
    days = st.slider("Show last N days", 7, 90, 30)
    recent = df.tail(days * 24)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=recent['timestamp'], y=recent['traffic_mbps'],
        fill='tozeroy', line=dict(color='#00d4ff', width=1.5),
        fillcolor='rgba(0,212,255,0.1)'
    ))
    fig.update_layout(
        paper_bgcolor='#0a0a1a', plot_bgcolor='#0a0a1a',
        font=dict(color='#8892b0'),
        xaxis=dict(gridcolor='#1a1a2e'),
        yaxis=dict(gridcolor='#1a1a2e', title='Traffic (Mbps)'),
        height=350, margin=dict(t=20)
    )
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🕐 Hourly Pattern")
        hourly = df.groupby('hour')['traffic_mbps'].mean().reset_index()
        fig2 = px.bar(hourly, x='hour', y='traffic_mbps',
                      color='traffic_mbps', color_continuous_scale='Blues')
        fig2.update_layout(paper_bgcolor='#0a0a1a', plot_bgcolor='#0a0a1a',
                           font=dict(color='#8892b0'), height=300,
                           xaxis=dict(gridcolor='#1a1a2e'),
                           yaxis=dict(gridcolor='#1a1a2e'),
                           coloraxis_showscale=False, margin=dict(t=10))
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        st.subheader("📅 Day of Week Pattern")
        day_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
        daily = df.groupby('day_of_week')['traffic_mbps'].mean().reindex(day_order).reset_index()
        fig3 = px.bar(daily, x='day_of_week', y='traffic_mbps',
                      color='traffic_mbps', color_continuous_scale='Teal')
        fig3.update_layout(paper_bgcolor='#0a0a1a', plot_bgcolor='#0a0a1a',
                           font=dict(color='#8892b0'), height=300,
                           xaxis=dict(gridcolor='#1a1a2e'),
                           yaxis=dict(gridcolor='#1a1a2e'),
                           coloraxis_showscale=False, margin=dict(t=10))
        st.plotly_chart(fig3, use_container_width=True)

# PAGE 2
elif page == "🔮 Model Predictions":
    st.title("🔮 Model Predictions")
    if not predictions:
        st.warning("Pehle models run karo!")
        st.stop()

    model_choice = st.selectbox("Model Select Karo", list(predictions.keys()))
    pred_df = predictions[model_choice]
    y_true = pred_df['actual'].values
    y_pred = pred_df.iloc[:, 1].values

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    col1, col2, col3 = st.columns(3)
    for col, (val, label) in zip([col1, col2, col3], [
        (f"{mae:.1f}", "MAE (Mbps)"),
        (f"{rmse:.1f}", "RMSE (Mbps)"),
        (f"{r2:.3f}", "R² Score")
    ]):
        with col:
            st.markdown(f"""<div class='metric-card'>
                <div class='metric-value'>{val}</div>
                <div class='metric-label'>{label}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    show_n = st.slider("Show N samples", 50, min(500, len(y_true)), 200)
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=y_true[:show_n], name='Actual',
                             line=dict(color='#00d4ff', width=2)))
    fig.add_trace(go.Scatter(y=y_pred[:show_n], name='Predicted',
                             line=dict(color='#ff6b6b', width=2, dash='dash')))
    fig.update_layout(
        paper_bgcolor='#0a0a1a', plot_bgcolor='#0a0a1a',
        font=dict(color='#8892b0'),
        xaxis=dict(gridcolor='#1a1a2e'),
        yaxis=dict(gridcolor='#1a1a2e', title='Traffic (Mbps)'),
        height=400, legend=dict(bgcolor='#1a1a2e')
    )
    st.plotly_chart(fig, use_container_width=True)

# PAGE 3
elif page == "📈 Model Comparison":
    st.title("📈 Model Comparison")
    if len(predictions) < 2:
        st.warning("Kam se kam 2 models run karo!")
        st.stop()

    comparison = []
    for name, pred_df in predictions.items():
        y_true = pred_df['actual'].values
        y_pred = pred_df.iloc[:, 1].values
        comparison.append({
            'Model': name,
            'MAE': round(mean_absolute_error(y_true, y_pred), 2),
            'RMSE': round(np.sqrt(mean_squared_error(y_true, y_pred)), 2),
            'R²': round(r2_score(y_true, y_pred), 4)
        })

    comp_df = pd.DataFrame(comparison).sort_values('RMSE')
    st.dataframe(comp_df, use_container_width=True, hide_index=True)

    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(comp_df, x='Model', y='RMSE', color='Model',
                     title='RMSE — Lower is Better',
                     color_discrete_sequence=['#00d4ff','#ff6b6b','#ffd166'])
        fig.update_layout(paper_bgcolor='#0a0a1a', plot_bgcolor='#0a0a1a',
                          font=dict(color='#8892b0'), showlegend=False,
                          yaxis=dict(gridcolor='#1a1a2e'), height=350)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.bar(comp_df, x='Model', y='R²', color='Model',
                     title='R² Score — Higher is Better',
                     color_discrete_sequence=['#00d4ff','#ff6b6b','#ffd166'])
        fig.update_layout(paper_bgcolor='#0a0a1a', plot_bgcolor='#0a0a1a',
                          font=dict(color='#8892b0'), showlegend=False,
                          yaxis=dict(gridcolor='#1a1a2e'), height=350)
        st.plotly_chart(fig, use_container_width=True)

# PAGE 4
elif page == "🚀 24Hr Forecast":
    st.title("🚀 Next 24 Hours Forecast")

    hourly_avg = df.groupby('hour')['traffic_mbps'].mean()
    hourly_std = df.groupby('hour')['traffic_mbps'].std()
    hours = list(range(24))
    forecast = [hourly_avg[h] for h in hours]
    upper = [hourly_avg[h] + hourly_std[h] for h in hours]
    lower = [hourly_avg[h] - hourly_std[h] for h in hours]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hours + hours[::-1], y=upper + lower[::-1],
        fill='toself', fillcolor='rgba(0,212,255,0.1)',
        line=dict(color='rgba(0,0,0,0)'), name='Confidence Band'
    ))
    fig.add_trace(go.Scatter(
        x=hours, y=forecast,
        line=dict(color='#00d4ff', width=3),
        mode='lines+markers', marker=dict(size=8),
        name='Forecast'
    ))
    fig.update_layout(
        paper_bgcolor='#0a0a1a', plot_bgcolor='#0a0a1a',
        font=dict(color='#8892b0'),
        xaxis=dict(gridcolor='#1a1a2e', title='Hour of Day',
                   tickvals=hours, ticktext=[f"{h}:00" for h in hours]),
        yaxis=dict(gridcolor='#1a1a2e', title='Traffic (Mbps)'),
        height=450, legend=dict(bgcolor='#1a1a2e')
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("⚡ Top 5 Peak Hours")
    peak_df = pd.DataFrame({
        'Hour': [f"{h}:00" for h in hours],
        'Forecast (Mbps)': [round(f, 1) for f in forecast]
    }).sort_values('Forecast (Mbps)', ascending=False).head(5)
    st.dataframe(peak_df, use_container_width=True, hide_index=True)