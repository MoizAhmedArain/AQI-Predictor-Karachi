import streamlit as st
import hopsworks
import plotly.graph_objects as go
import os
import pandas as pd
from dotenv import load_dotenv
from ui_component import *

#  PAGE CONFIG 
st.set_page_config(
    page_title="Karachi Air Quality",
    layout="wide"
)

#  LOAD CSS 
def load_css():
    with open("AQI_app/styles.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

#  PARTICLE BACKGROUND 
st.markdown("""
<script src="https://cdn.jsdelivr.net/npm/particles.js@2.0.0/particles.min.js"></script>
<div id="particles-js"></div>

<script>
particlesJS('particles-js', {
  particles: {
    number: { value: 60 },
    color: { value: "#6b7280" },
    opacity: { value: 0.3 },
    size: { value: 3 },
    move: { enable: true, speed: 1 }
  }
});
</script>
""", unsafe_allow_html=True)

load_dotenv()

HOPSWORKS_KEY = os.getenv("HOPSWORKS_API_KEY")
HOPSWORKS_PROJECT = os.getenv("HOPSWORKS_PROJECT_NAME")

@st.cache_resource
def get_project():
    return hopsworks.login(
        api_key_value=HOPSWORKS_KEY,
        project=HOPSWORKS_PROJECT
    )


#  DATA 
@st.cache_data(ttl=3600)
def get_data():
        project = get_project()
        fs = project.get_feature_store()
        fg = fs.get_feature_group(name="aqi_predictions", version=1)

        df = fg.read()
        df = df.sort_values("prediction_time").tail(72)

        return df

df = get_data()
latest_aqi = float(df["predicted_pm2_5"].iloc[-1])

#  HEADER 
st.title(" Karachi Air Intelligence")
st.caption("AI Atmospheric Forecasting Platform")

#  KPI 
col1, col2, col3 = st.columns(3)

status_text, status_class = aqi_status(latest_aqi)

with col1:
    glass_card_start()
    st.metric("Current PM2.5", f"{latest_aqi:.2f} μg/m³")
    glass_card_end()

with col2:
    glass_card_start()
    st.metric("72H Peak", f"{df['predicted_pm2_5'].max():.2f} μg/m³")
    glass_card_end()

with col3:
    glass_card_start()
    st.markdown(
        f"<h4>Status</h4><span class='{status_class}' style='font-size:28px'>{status_text}</span>",
        unsafe_allow_html=True
    )
    glass_card_end()

#  CHART 
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=df["prediction_time"],
    y=df["predicted_pm2_5"],
    mode="lines",
    name="PM2.5 Forecast",
    line=dict(width=3)
))

fig.update_layout(
    title="72 Hour PM2.5 Forecast",
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#e6edf3"),
    hovermode="x unified",
    height=500
)

st.plotly_chart(fig, use_container_width=True)

#  MODEL INFO 
with st.expander("🔬 Model Engineering"):

    c1, c2, c3 = st.columns(3)

    c1.write("Model")
    c1.code("Random Forest v2")

    c2.write("RMSE")
    c2.code("4.19")

    c3.write("Forecast Type")
    c3.code("Recursive Multi-Step")


  # 1. FETCH HISTORICAL DATA FOR EDA (With error handling)
@st.cache_data(ttl=3600)
def get_historical_eda():
    try:
        project = hopsworks.login() # Or your get_project() function
        fs = project.get_feature_store()
        # Ensure the name matches your Hopsworks Feature Group exactly
        aqi_fg = fs.get_feature_group(name="karachi_aqi_weather", version=1)
        df = aqi_fg.read()
        
        # Smart Check: If 'hour' isn't a column, extract it from 'time'
        if 'hour' not in df.columns and 'time' in df.columns:
            df['time'] = pd.to_datetime(df['time'])
            df['hour'] = df['time'].dt.hour
            
        return df
    except Exception as e:
        st.error(f"EDA Data Load Error: {e}")
        return None

eda_df = get_historical_eda()

if eda_df is not None:
    col_eda1, col_eda2 = st.columns(2)

    with col_eda1:
        # --- DIURNAL CYCLE (The Heartbeat) ---
        st.subheader(" The Daily Heartbeat")
        # Ensure we use pm2_5 (underscore)
        hourly_avg = eda_df.groupby('hour')['pm2_5'].mean().reset_index()
        
        fig_hour = go.Figure()
        fig_hour.add_trace(go.Scatter(
            x=hourly_avg['hour'], 
            y=hourly_avg['pm2_5'], 
            fill='tozeroy', 
            line_color='#00d4ff',
            name="Avg PM2.5"
        ))
        fig_hour.update_layout(
            title="Hourly Pollution Patterns (Karachi)",
            xaxis_title="Hour of Day (24h)",
            yaxis_title="PM2.5 (µg/m³)",
            paper_bgcolor='rgba(0,0,0,0)', 
            plot_bgcolor='rgba(0,0,0,0)', 
            font=dict(color='white'),
            height=400
        )
        st.plotly_chart(fig_hour, use_container_width=True)
        st.info("**Insight:** Notice peaks during 8 AM - 10 AM and 8 PM - 11 PM, matching Karachi's traffic and cooling air.")

    with col_eda2:
        # --- WIND IMPACT (The Sea Breeze) ---
        st.subheader(" The Sea Breeze Effect")
        fig_wind = go.Figure()
        fig_wind.add_trace(go.Scatter(
            x=eda_df['wind_speed_10m'], 
            y=eda_df['pm2_5'], 
            mode='markers', 
            marker=dict(color='#00d4ff', opacity=0.4, size=8),
            name="Observations"
        ))
        fig_wind.update_layout(
            title="Wind Speed vs. Air Quality", 
            xaxis_title="Wind Speed (km/h)", 
            yaxis_title="PM2.5 (µg/m³)", 
            paper_bgcolor='rgba(0,0,0,0)', 
            plot_bgcolor='rgba(0,0,0,0)', 
            font=dict(color='white'),
            height=400
        )
        st.plotly_chart(fig_wind, use_container_width=True)
        st.info(" **Insight:** High wind speeds (>15 km/h) consistently result in lower PM2.5 levels.")

    # --- CORRELATION HEATMAP ---
    st.subheader(" Feature Interaction Matrix")
    # Using correct column names for the heatmap
    cols_to_corr = ['pm2_5', 'temperature_2m', 'relative_humidity_2m', 'wind_speed_10m']
    
    # Filter only columns that actually exist in df to prevent crashes
    existing_cols = [c for c in cols_to_corr if c in eda_df.columns]
    corr = eda_df[existing_cols].corr()

    fig_heat = go.Figure(data=go.Heatmap(
                z=corr.values,
                x=corr.columns,
                y=corr.columns,
                colorscale='Viridis',
                zmin=-1, zmax=1))
    
    fig_heat.update_layout(
        title="Meteorological Correlation with Pollution", 
        paper_bgcolor='rgba(0,0,0,0)', 
        plot_bgcolor='rgba(0,0,0,0)', 
        font=dict(color='white'),
        height=500
    )
    st.plotly_chart(fig_heat, use_container_width=True)
else:
    st.warning(" Waiting for Historical Feature Group to load...")