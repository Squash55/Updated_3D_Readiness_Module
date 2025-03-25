
import streamlit as st
import pandas as pd
import pydeck as pdk

st.set_page_config(page_title="Readiness Map by Quartile (Artificial data)", layout="wide")

@st.cache_data
def load_data():
    return pd.read_csv("USAF_100_Base_Data.csv")

df = load_data()

st.title("🗺️ Mission Readiness Map by Quartile (Artificial data)")
st.markdown("Bases are color-coded by quartile of readiness score (green = highest, red = lowest).")

# Compute quartiles
q1 = df["Readiness"].quantile(0.25)
q2 = df["Readiness"].quantile(0.50)
q3 = df["Readiness"].quantile(0.75)

def assign_color(readiness):
    if readiness <= q1:
        return [255, 0, 0]       # Red
    elif readiness <= q2:
        return [255, 165, 0]     # Orange
    elif readiness <= q3:
        return [255, 255, 0]     # Yellow
    else:
        return [0, 200, 0]       # Green

df["color"] = df["Readiness"].apply(assign_color)

# Map layer
layer = pdk.Layer(
    "ScatterplotLayer",
    data=df,
    get_position="[Longitude, Latitude]",
    get_color="color",
    get_radius=30000,
    pickable=True,
)

# View settings
view_state = pdk.ViewState(
    latitude=df["Latitude"].mean(),
    longitude=df["Longitude"].mean(),
    zoom=4,
    pitch=30,
)

st.pydeck_chart(pdk.Deck(
    layers=[layer],
    initial_view_state=view_state,
    tooltip={"text": "Base: {Base}\nReadiness: {Readiness}"}
))

# Insight summary
st.subheader("📊 Quartile-Based Readiness Summary")
st.markdown(f"- 🟥 **Q1 (Lowest Readiness ≤ {q1:.1f})**")
st.markdown(f"- 🟧 **Q2 (≤ {q2:.1f})**")
st.markdown(f"- 🟨 **Q3 (≤ {q3:.1f})**")
st.markdown("- 🟩 **Q4 (Highest Readiness)**")
st.markdown("Use color patterns to locate low-readiness clusters and strategic performance zones.")
