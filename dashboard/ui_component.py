import streamlit as st

def glass_card_start():
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)

def glass_card_end():
    st.markdown('</div>', unsafe_allow_html=True)

def aqi_status(aqi):
    if aqi <= 50:
        return "Good", "aqi-good"
    elif aqi <= 100:
        return "Moderate", "aqi-moderate"
    elif aqi <= 200:
        return "Unhealthy", "aqi-unhealthy"
    else:
        return "Hazardous", "aqi-hazard"
