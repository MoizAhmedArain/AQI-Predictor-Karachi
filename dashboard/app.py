import streamlit as st
import hopsworks
import plotly.graph_objects as go
import os
import pandas as pd
from dotenv import load_dotenv
from ui_component import * 


st.set_page_config(
    page_title="Karachi Air Intelligence",
    layout="wide"
)


def load_css():
    try:
        with open("dashboard/styles.css") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning("CSS file not found. Check path: dashboard/styles.css")

load_css()

st.markdown("""
<div id="particles-js" style="position: fixed; top: 0; left: 0; width: 100%; height: 100%; z-index: -1;"></div>
<script src="https://cdn.jsdelivr.net/npm/particles.js@2.0.0/particles.min.js"></script>
<script>
particlesJS('particles-js', {
  particles: {
    number: { value: 50 },
    color: { value: "#00d4ff" },
    opacity: { value: 0.2 },
    size: { value: 2 },
    line_linked: { enable: true, distance: 150, color: "#00d4ff", opacity: 0.1, width: 1 },
    move: { enable: true, speed: 1 }
  }
});
</script>
""", unsafe_allow_html=True)

load_dotenv()

try:
    @st.cache_resource
    def get_project():
        try:
            return hopsworks.login(
                api_key_value=os.getenv("HOPSWORKS_API_KEY"),
                project=os.getenv("HOPSWORKS_PROJECT_NAME")
            )
        except Exception as e:
            st.error(f"Failed to login to Hopsworks: {e}")
            raise e

    @st.cache_data(ttl=3600)
    def get_forecast_data():
        try:
            project = get_project()
            fs = project.get_feature_store()
            fg = fs.get_feature_group(name="aqi_predictions", version=1)
            df = fg.read()
            
            # CRITICAL: Ensure time is datetime for the graph
            df["prediction_time"] = pd.to_datetime(df["prediction_time"])
            df = df.sort_values("prediction_time")
            return df
        except Exception as e:
            st.error(f"Failed to load forecast data: {e}")
    

    # fetching the historical data
    @st.cache_data(ttl=3600)
    def get_historical_eda():
        try:
            project = get_project()
            fs = project.get_feature_store()
            aqi_fg = fs.get_feature_group(name="karachi_aqi_weather", version=1)
            df = aqi_fg.read()
            if 'hour' not in df.columns and 'time' in df.columns:
                df['time'] = pd.to_datetime(df['time'])
                df['hour'] = df['time'].dt.hour
            return df
        except Exception as e:
            st.error(f"EDA Data Load Error: {e}")
            return None

    df_forecast = get_forecast_data()
    eda_df = get_historical_eda()

    latest_aqi = float(df_forecast["predicted_pm2_5"].iloc[0])

    st.title(" Karachi Air Intelligence")
    st.markdown("<p style='opacity:0.7; font-size:1.2rem;'>AI Atmospheric Forecasting Platform</p>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    status_text, status_class = aqi_status(latest_aqi)

    with col1:
        glass_card_start()
        st.metric("Current PM2.5 (Est.)", f"{latest_aqi:.1f} μg/m³")
        st.caption("Immediate forecast for the current hour")
        glass_card_end()

    with col2:
        glass_card_start()
        peak_val = df_forecast['predicted_pm2_5'].max()
        st.metric("72H Expected Peak", f"{peak_val:.1f} μg/m³")
        st.caption("Maximum pollution level in next 3 days")
        glass_card_end()

    with col3:
        glass_card_start()
        st.markdown(f"<h4>Atmospheric Status</h4>", unsafe_allow_html=True)
        st.markdown(f"<div class='{status_class}' style='font-size:24px; font-weight:bold;'>{status_text}</div>", unsafe_allow_html=True)
        glass_card_end()

    
    st.markdown("### 72 Hour Trajectory") 
    if st.button("Clear cache"):
        st.cache_data.clear()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_forecast["prediction_time"],
        y=df_forecast["predicted_pm2_5"],
        mode="lines+markers",
        name="PM2.5 Forecast",
        line=dict(color='#00d4ff', width=3),
        fill='tozeroy',
        fillcolor='rgba(0, 212, 255, 0.1)'
    ))

    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e6edf3"),
        hovermode="x unified",
        margin=dict(l=0, r=0, t=30, b=0),
        height=400,
        xaxis=dict(showgrid=False),
        yaxis=dict(gridcolor="rgba(255,255,255,0.05)", title="μg/m³")
    )
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.markdown("### Historical Insights (City Patterns)")

    if eda_df is not None:
        tab1, tab2 = st.tabs(["Daily Cycle", "Weather Correlations"])
        
        with tab1:
            hourly_avg = eda_df.groupby('hour')['pm2_5'].mean().reset_index()
            fig_hour = go.Figure(go.Scatter(x=hourly_avg['hour'], y=hourly_avg['pm2_5'], fill='tozeroy', line_color='#00d4ff'))
            fig_hour.update_layout(title="Average PM2.5 by Hour (Local Time)", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
            st.plotly_chart(fig_hour, use_container_width=True)

        with tab2:
            cols = ['pm2_5', 'temperature_2m', 'relative_humidity_2m', 'wind_speed_10m']
            corr = eda_df[cols].corr()
            fig_heat = go.Figure(data=go.Heatmap(z=corr.values, x=corr.columns, y=corr.columns, colorscale='Viridis'))
            fig_heat.update_layout(title="Correlation: Weather vs Pollution", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
            st.plotly_chart(fig_heat, use_container_width=True)

    with st.expander(" Model Engineering & Metadata"):
        c1, c2, c3 = st.columns(3)
        c1.info("**Algorithm:** Random Forest v2")
        c2.info("**Training RMSE:** 4.19")
        c3.info("**Strategy:** Recursive Multi-Step")

except Exception as e:
    st.error(f"Dashboard error: {e}")
