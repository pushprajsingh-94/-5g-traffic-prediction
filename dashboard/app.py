import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os, sys, joblib
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(page_title="5G Traffic Predictor", page_icon="📡", layout="wide")

st.markdown("""
<style>
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
    .alert-box {
        background: linear-gradient(135deg, #2d1b1b, #3d1f1f);
        border: 1px solid #ff4444;
        border-radius: 12px;
        padding: 15px;
        margin: 5px;
    }
    .normal-box {
        background: linear-gradient(135deg, #1b2d1b, #1f3d1f);
        border: 1px solid #44ff44;
        border-radius: 12px;
        padding: 15px;
        margin: 5px;
    }
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

@st.cache_resource
def load_rf_model():
    if os.path.exists('saved_models/rf_model.pkl'):
        return joblib.load('saved_models/rf_model.pkl')
    return None

df = load_traffic_data()
predictions = load_predictions()
rf_model = load_rf_model()

st.sidebar.markdown("## 📡 5G Traffic Predictor")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigate", [
    "📊 Data Overview",
    "🔮 Model Predictions",
    "📈 Model Comparison",
    "🚨 Anomaly Detection",
    "🎯 Live Predictor",
    "🔥 Advanced Analysis",
    "🚀 24Hr Forecast"
])
st.sidebar.markdown("---")
st.sidebar.markdown("**By:** Pushpraj Singh")
st.sidebar.markdown("**Stack:** Python · Random Forest · ARIMA · Linear Regression")

# ── PAGE 1: DATA OVERVIEW ──────────────────────────────────────
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
    fig.add_trace(go.Scatter(x=recent['timestamp'], y=recent['traffic_mbps'],
        fill='tozeroy', line=dict(color='#00d4ff', width=1.5),
        fillcolor='rgba(0,212,255,0.1)'))
    fig.update_layout(paper_bgcolor='#0a0a1a', plot_bgcolor='#0a0a1a',
        font=dict(color='#8892b0'), xaxis=dict(gridcolor='#1a1a2e'),
        yaxis=dict(gridcolor='#1a1a2e', title='Traffic (Mbps)'),
        height=350, margin=dict(t=20))
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

# ── PAGE 2: MODEL PREDICTIONS ──────────────────────────────────
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
        (f"{mae:.1f}", "MAE (Mbps)"), (f"{rmse:.1f}", "RMSE (Mbps)"), (f"{r2:.3f}", "R² Score")]):
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
    fig.update_layout(paper_bgcolor='#0a0a1a', plot_bgcolor='#0a0a1a',
        font=dict(color='#8892b0'), xaxis=dict(gridcolor='#1a1a2e'),
        yaxis=dict(gridcolor='#1a1a2e', title='Traffic (Mbps)'),
        height=400, legend=dict(bgcolor='#1a1a2e'))
    st.plotly_chart(fig, use_container_width=True)

# ── PAGE 3: MODEL COMPARISON ───────────────────────────────────
elif page == "📈 Model Comparison":
    st.title("📈 Model Comparison")
    if len(predictions) < 2:
        st.warning("Kam se kam 2 models run karo!")
        st.stop()
    comparison = []
    for name, pred_df in predictions.items():
        y_true = pred_df['actual'].values
        y_pred = pred_df.iloc[:, 1].values
        comparison.append({'Model': name,
            'MAE': round(mean_absolute_error(y_true, y_pred), 2),
            'RMSE': round(np.sqrt(mean_squared_error(y_true, y_pred)), 2),
            'R²': round(r2_score(y_true, y_pred), 4)})
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

# ── PAGE 4: ANOMALY DETECTION ──────────────────────────────────
elif page == "🚨 Anomaly Detection":
    st.title("🚨 Anomaly Detection")
    st.markdown("Unusual traffic spikes automatically detect karta hai!")

    days = st.slider("Analyze last N days", 7, 60, 30)
    recent = df.tail(days * 24).copy()

    mean = recent['traffic_mbps'].mean()
    std = recent['traffic_mbps'].std()
    threshold = st.slider("Sensitivity (Standard Deviations)", 1.5, 3.0, 2.0, 0.1)

    recent['z_score'] = (recent['traffic_mbps'] - mean) / std
    recent['is_anomaly'] = recent['z_score'].abs() > threshold
    anomalies = recent[recent['is_anomaly']]

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-value'>{len(anomalies)}</div>
            <div class='metric-label'>Anomalies Detected</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-value'>{mean:.0f}</div>
            <div class='metric-label'>Mean Traffic (Mbps)</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        pct = (len(anomalies)/len(recent)*100)
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-value'>{pct:.1f}%</div>
            <div class='metric-label'>Anomaly Rate</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=recent['timestamp'], y=recent['traffic_mbps'],
        line=dict(color='#00d4ff', width=1), name='Traffic'))
    fig.add_trace(go.Scatter(
        x=[recent['timestamp'].iloc[0], recent['timestamp'].iloc[-1]],
        y=[mean + threshold*std, mean + threshold*std],
        line=dict(color='#ff4444', dash='dash', width=1), name='Upper Threshold'))
    fig.add_trace(go.Scatter(
        x=[recent['timestamp'].iloc[0], recent['timestamp'].iloc[-1]],
        y=[mean - threshold*std, mean - threshold*std],
        line=dict(color='#ff4444', dash='dash', width=1), name='Lower Threshold'))
    if len(anomalies) > 0:
        fig.add_trace(go.Scatter(x=anomalies['timestamp'], y=anomalies['traffic_mbps'],
            mode='markers', marker=dict(color='#ff4444', size=10, symbol='x'),
            name='Anomaly'))
    fig.update_layout(paper_bgcolor='#0a0a1a', plot_bgcolor='#0a0a1a',
        font=dict(color='#8892b0'), xaxis=dict(gridcolor='#1a1a2e'),
        yaxis=dict(gridcolor='#1a1a2e', title='Traffic (Mbps)'),
        height=400, legend=dict(bgcolor='#1a1a2e'))
    st.plotly_chart(fig, use_container_width=True)

    if len(anomalies) > 0:
        st.subheader("🔴 Detected Anomalies")
        st.dataframe(anomalies[['timestamp','traffic_mbps','z_score']].round(2),
                     use_container_width=True, hide_index=True)

# ── PAGE 5: LIVE PREDICTOR ─────────────────────────────────────
elif page == "🎯 Live Predictor":
    st.title("🎯 Live Traffic Predictor")
    st.markdown("Apni values daalo — model turant predict karega!")

    col1, col2, col3 = st.columns(3)
    with col1:
        hour = st.selectbox("Hour of Day", list(range(24)),
                           format_func=lambda x: f"{x}:00")
    with col2:
        day = st.selectbox("Day of Week", 
                          ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'])
    with col3:
        month = st.selectbox("Month", 
                            ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])

    day_map = {'Monday':0,'Tuesday':1,'Wednesday':2,'Thursday':3,'Friday':4,'Saturday':5,'Sunday':6}
    month_map = {'Jan':1,'Feb':2,'Mar':3,'Apr':4,'May':5,'Jun':6,'Jul':7,'Aug':8,'Sep':9,'Oct':10,'Nov':11,'Dec':12}
    is_weekend = 1 if day in ['Saturday','Sunday'] else 0

    hourly_avg = df.groupby('hour')['traffic_mbps'].mean()
    lag_1 = hourly_avg[hour]
    lag_24 = hourly_avg[(hour - 24) % 24]
    lag_168 = hourly_avg[hour]
    rolling_mean_6 = df['traffic_mbps'].rolling(6).mean().mean()
    rolling_mean_24 = df['traffic_mbps'].rolling(24).mean().mean()

    features = [[hour, day_map[day], month_map[month], is_weekend,
                 lag_1, lag_24, lag_168, rolling_mean_6, rolling_mean_24]]

    if st.button("🚀 Predict Traffic!", use_container_width=True):
        if rf_model:
            prediction = rf_model.predict(features)[0]
            std = df['traffic_mbps'].std()

            st.markdown("<br>", unsafe_allow_html=True)
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"""<div class='metric-card'>
                    <div class='metric-value'>{prediction:.0f}</div>
                    <div class='metric-label'>Predicted Traffic (Mbps)</div>
                </div>""", unsafe_allow_html=True)
            with col2:
                st.markdown(f"""<div class='metric-card'>
                    <div class='metric-value'>{prediction-std*0.5:.0f} - {prediction+std*0.5:.0f}</div>
                    <div class='metric-label'>Confidence Range (Mbps)</div>
                </div>""", unsafe_allow_html=True)
            with col3:
                overall_avg = df['traffic_mbps'].mean()
                diff = ((prediction - overall_avg) / overall_avg) * 100
                color = "🔴 HIGH" if diff > 20 else "🟡 MEDIUM" if diff > 0 else "🟢 LOW"
                st.markdown(f"""<div class='metric-card'>
                    <div class='metric-value'>{color}</div>
                    <div class='metric-label'>Traffic Level</div>
                </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            if diff > 20:
                st.markdown("""<div class='alert-box'>
                    🚨 <b>HIGH TRAFFIC ALERT!</b> Network resources badhao — congestion ho sakta hai!
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown("""<div class='normal-box'>
                    ✅ <b>Normal Traffic</b> — Network stable hai, koi action required nahi!
                </div>""", unsafe_allow_html=True)

            # Gauge chart
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prediction,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Predicted Traffic (Mbps)", 'font': {'color': '#8892b0'}},
                gauge={
                    'axis': {'range': [0, 2000], 'tickcolor': '#8892b0'},
                    'bar': {'color': '#00d4ff'},
                    'steps': [
                        {'range': [0, 600], 'color': '#1a1a2e'},
                        {'range': [600, 1200], 'color': '#16213e'},
                        {'range': [1200, 2000], 'color': '#0f3460'}],
                    'threshold': {'line': {'color': '#ff4444', 'width': 4},
                                  'thickness': 0.75, 'value': 1500}}))
            fig.update_layout(paper_bgcolor='#0a0a1a', font=dict(color='#8892b0'), height=350)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("Model file nahi mili! Pehle random_forest_model.py run karo.")

# ── PAGE 6: ADVANCED ANALYSIS ──────────────────────────────────
elif page == "🔥 Advanced Analysis":
    st.title("🔥 Advanced Analysis")

    tab1, tab2, tab3 = st.tabs(["🗺️ Heatmap", "📊 Distribution", "🔄 What-If Analysis"])

    with tab1:
        st.subheader("Traffic Heatmap — Hour vs Day")
        heatmap_data = df.groupby(['hour', 'day_of_week'])['traffic_mbps'].mean().reset_index()
        heatmap_pivot = heatmap_data.pivot(index='day_of_week', columns='hour', values='traffic_mbps')
        day_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
        heatmap_pivot = heatmap_pivot.reindex(day_order)
        fig = px.imshow(heatmap_pivot, color_continuous_scale='Blues',
                        labels=dict(x="Hour of Day", y="Day", color="Mbps"))
        fig.update_layout(paper_bgcolor='#0a0a1a', plot_bgcolor='#0a0a1a',
                          font=dict(color='#8892b0'), height=400)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("Traffic Distribution")
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=df['traffic_mbps'], nbinsx=50,
                                   marker_color='#00d4ff', opacity=0.7))
        fig.update_layout(paper_bgcolor='#0a0a1a', plot_bgcolor='#0a0a1a',
                          font=dict(color='#8892b0'), height=400,
                          xaxis=dict(gridcolor='#1a1a2e', title='Traffic (Mbps)'),
                          yaxis=dict(gridcolor='#1a1a2e', title='Frequency'))
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.subheader("⚡ What-If Analysis")
        st.markdown("Agar conditions change ho toh traffic kya hoga?")
        col1, col2 = st.columns(2)
        with col1:
            scenario = st.selectbox("Scenario Select Karo", [
                "Weekday vs Weekend",
                "Morning vs Evening Peak",
                "Summer vs Winter"
            ])
        with col2:
            if scenario == "Weekday vs Weekend":
                weekday = df[df['is_weekend']==0]['traffic_mbps'].mean()
                weekend = df[df['is_weekend']==1]['traffic_mbps'].mean()
                comp = pd.DataFrame({'Type': ['Weekday', 'Weekend'], 'Avg Traffic': [weekday, weekend]})
            elif scenario == "Morning vs Evening Peak":
                morning = df[df['hour'].between(8,11)]['traffic_mbps'].mean()
                evening = df[df['hour'].between(19,22)]['traffic_mbps'].mean()
                comp = pd.DataFrame({'Type': ['Morning (8-11)', 'Evening (19-22)'], 'Avg Traffic': [morning, evening]})
            else:
                summer = df[df['month'].between(4,6)]['traffic_mbps'].mean()
                winter = df[df['month'].between(11,12)]['traffic_mbps'].mean()
                comp = pd.DataFrame({'Type': ['Summer', 'Winter'], 'Avg Traffic': [summer, winter]})

            fig = px.bar(comp, x='Type', y='Avg Traffic',
                         color='Type', color_discrete_sequence=['#00d4ff','#ff6b6b'])
            fig.update_layout(paper_bgcolor='#0a0a1a', plot_bgcolor='#0a0a1a',
                              font=dict(color='#8892b0'), showlegend=False,
                              yaxis=dict(gridcolor='#1a1a2e'), height=300)
            st.plotly_chart(fig, use_container_width=True)

# ── PAGE 7: 24HR FORECAST ──────────────────────────────────────
elif page == "🚀 24Hr Forecast":
    st.title("🚀 Next 24 Hours Forecast")
    hourly_avg = df.groupby('hour')['traffic_mbps'].mean()
    hourly_std = df.groupby('hour')['traffic_mbps'].std()
    hours = list(range(24))
    forecast = [hourly_avg[h] for h in hours]
    upper = [hourly_avg[h] + hourly_std[h] for h in hours]
    lower = [hourly_avg[h] - hourly_std[h] for h in hours]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hours + hours[::-1], y=upper + lower[::-1],
        fill='toself', fillcolor='rgba(0,212,255,0.1)',
        line=dict(color='rgba(0,0,0,0)'), name='Confidence Band'))
    fig.add_trace(go.Scatter(x=hours, y=forecast,
        line=dict(color='#00d4ff', width=3),
        mode='lines+markers', marker=dict(size=8), name='Forecast'))
    fig.update_layout(paper_bgcolor='#0a0a1a', plot_bgcolor='#0a0a1a',
        font=dict(color='#8892b0'),
        xaxis=dict(gridcolor='#1a1a2e', title='Hour of Day',
                   tickvals=hours, ticktext=[f"{h}:00" for h in hours]),
        yaxis=dict(gridcolor='#1a1a2e', title='Traffic (Mbps)'),
        height=450, legend=dict(bgcolor='#1a1a2e'), margin=dict(t=10))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("⚡ Top 5 Peak Hours")
    peak_df = pd.DataFrame({
        'Hour': [f"{h}:00" for h in hours],
        'Forecast (Mbps)': [round(f, 1) for f in forecast]
    }).sort_values('Forecast (Mbps)', ascending=False).head(5)
    st.dataframe(peak_df, use_container_width=True, hide_index=True)

    # Download button
    csv = peak_df.to_csv(index=False)
    st.download_button("📥 Download Forecast CSV", csv, "forecast.csv", "text/csv")