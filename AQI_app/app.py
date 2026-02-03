import streamlit as st
import hopsworks
import plotly.graph_objects as go
import os
from dotenv import load_dotenv
from ui_component import *

#  PAGE CONFIG 
st.set_page_config(
    page_title="Karachi Air Intel",
    page_icon="🌬️",
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

#  ENV LOAD 
load_dotenv()

HOPSWORKS_KEY = os.getenv("HOPSWORKS_API_KEY")
HOPSWORKS_PROJECT = os.getenv("HOPSWORKS_PROJECT_NAME")

#  DATA 
@st.cache_data(ttl=3600)
def get_data():

    if not HOPSWORKS_KEY:
        st.error("Missing Hopsworks API Key")
        st.stop()

    with st.spinner("Connecting to Air Intelligence Center..."):

        project = hopsworks.login(
            api_key_value=HOPSWORKS_KEY,
            project=HOPSWORKS_PROJECT
        )

        fs = project.get_feature_store()
        fg = fs.get_feature_group(name="aqi_predictions", version=1)

        df = fg.read()
        df = df.sort_values("prediction_time").tail(72)

        return df

df = get_data()
latest_aqi = float(df["predicted_pm2_5"].iloc[-1])

#  HEADER 
st.title("🌬️ Karachi Air Intelligence")
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
