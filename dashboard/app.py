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

# --- SIDEBAR & REFRESH LOGIC ---
st.sidebar.title("Control Panel")
if st.sidebar.button('Refresh Live Data'):
    st.cache_data.clear()
    st.rerun()

st.sidebar.write("---")
st.sidebar.subheader("Project Maturity: Level 3")
st.sidebar.info("System Version: `AQI-KHI-v2.1.0` (Stable)")

# Bar-- of Aqi simple

# Hero Section
st.markdown("""
    <style>
    .hero {
        background: linear-gradient(90deg, #00BCD4, #2196F3);
        padding: 50px 20px;
        border-radius: 20px;
        text-align: center;
        color: white;
        margin-bottom: 30px;
    }
    .hero h1 {
        font-size: 48px;
        font-weight: 700;
        margin-bottom: 10px;
    }
    .hero p {
        font-size: 20px;
        opacity: 0.9;
    }
    </style>

    <div class="hero">
        <h1>Air Quality Intelligence Dashboard</h1>
        <p>Real time AQI Monitoring & AI Powered 3-Day Forecast</p>
    </div>
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
            df = df.sort_values("prediction_time", ascending=False)
            latest_batch = df.head(72).sort_values("prediction_time")
            return latest_batch
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
    # if st.button("Clear cache"):
    #     st.cache_data.clear()
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
        font=dict(color="#111212"),
        hovermode="x unified",
        margin=dict(l=0, r=0, t=30, b=0),
        height=400,
        xaxis=dict(showgrid=False),
        yaxis=dict(gridcolor="rgba(255,255,255,0.05)", title="μg/m³")
    )
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.markdown("### Historical Insights Karachi")

    if eda_df is not None:
        tab1, tab2 = st.tabs(["Daily Cycle", "Weather Correlations"])
        
        with tab1:
            hourly_avg = eda_df.groupby('hour')['pm2_5'].mean().reset_index()
            fig_hour = go.Figure(go.Scatter(x=hourly_avg['hour'], y=hourly_avg['pm2_5'], fill='tozeroy', line_color="#090909"))
            fig_hour.update_layout(title="Average PM2.5 by Hour (Local Time)", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
            st.plotly_chart(fig_hour, use_container_width=True)

        with tab2:
            cols = ['pm2_5', 'temperature_2m', 'relative_humidity_2m', 'wind_speed_10m']
            corr = eda_df[cols].corr()
            fig_heat = go.Figure(data=go.Heatmap(z=corr.values, x=corr.columns, y=corr.columns, colorscale='Viridis'))
            fig_heat.update_layout(title="Correlation: Weather vs Pollution", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
            st.plotly_chart(fig_heat, use_container_width=True)

    # 2. 
    st.write("---")
    st.subheader("Daily Forecast Summary")
    
    df_daily = df_forecast.copy()
    df_daily['date'] = df_daily['prediction_time'].dt.strftime('%a, %d %b')
    daily_stats = df_daily.groupby('date')['predicted_pm2_5'].agg(['mean', 'min', 'max']).reset_index()
    daily_stats['sort_date'] = pd.to_datetime(daily_stats['date'], format='%a, %d %b')
    daily_stats = daily_stats.sort_values('sort_date').head(3)

    
   
    card_cols = st.columns(3, gap="medium")

    st.markdown("""
        <style>
        .forecast-card {
            /* Matching the deep charcoal/navy from your image */
            background-color: ##ffffff; 
            border-radius: 12px; 
            
            /* Increasing vertical size */
            padding: 40px 20px; 
            min-height: 220px;
            
            text-align: center;
            
            /* The specific orange bottom border from your screenshot */
            border-bottom: 6px solid #f6742a; 
            
            margin-bottom: 20px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.3);
        }
        
        .date-text { 
            color: #21252e; 
            font-size: 1.3rem; 
            font-weight: 600; 
            margin-bottom: 15px;
        }
        
        .aqi-val { 
            /* Vibrant orange for the main number */
            color: #ffffff; 
            font-size: 3.5rem; 
            font-weight: 800; 
            line-height: 1;
        }
        
        .range-text { 
            color: #ffffff; 
            font-size: 0.9rem; 
            margin-top: 10px;
            letter-spacing: 0.5px;
        }
        </style>
    """, unsafe_allow_html=True)

    # Loop to render cards
    for i, (_, row) in enumerate(daily_stats.iterrows()):
        with card_cols[i]:
            st.markdown(f"""
                <div class="forecast-card">
                    <div class="date-text">{row['date']}</div>
                    <div class="aqi-val">{int(row['mean'])}</div>
                    <div class="range-text">Range: {int(row['min'])} - {int(row['max'])}</div>
                </div>
            """, unsafe_allow_html=True)
    
    with st.expander(" Model Engineering & Metadata"):
    # Custom CSS for the Black Metadata Cards
        st.markdown("""
            <style>
            .meta-card {
                background-color: #ffffff; /* Pure White background */
                border: 1px solid #333333; /* Dark grey border for definition */
                border-radius: 10px;
                padding: 20px;
                text-align: center;
                margin-bottom: 10px;
            }
            .meta-label {
                color: #000000; /* black label */
                font-size: 0.85rem;
                text-transform: uppercase;
                letter-spacing: 1px;
            }
            .meta-value {
                color: #000000; /* Black text */
                font-size: 1.1rem;
                font-weight: 600;
                margin-top: 5px;
            }
            </style>
        """, unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)
        
        with c1:
            st.markdown("""<div class="meta-card">
                <div class="meta-label">Algorithm</div>
                <div class="meta-value">Random Forest v2</div>
            </div>""", unsafe_allow_html=True)
            
        with c2:
            st.markdown("""<div class="meta-card">
                <div class="meta-label">Training RMSE</div>
                <div class="meta-value">4.19</div>
            </div>""", unsafe_allow_html=True)
            
        with c3:
            st.markdown("""<div class="meta-card">
                <div class="meta-label">Strategy</div>
                <div class="meta-value">Recursive Multi-Step</div>
            </div>""", unsafe_allow_html=True)

except Exception as e:
    st.error(f"Dashboard error: {e}")
