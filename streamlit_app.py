
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
import numpy as np

st.set_page_config(page_title="3D Mission Readiness Surface (Artificial data)", layout="wide")

@st.cache_data
def load_data():
    return pd.read_csv("USAF_3D_Data.csv")

df = load_data()

st.title("🛰️ 3D Mission Readiness Surface (Artificial data)")
st.markdown("This chart visualizes how Mission Complexity and Maintenance Burden influence Readiness Score.")

# Setup axes
x_col = "Mission Complexity"
y_col = "Maintenance Burden"
z_col = "Readiness Score"

x = df[x_col]
y = df[y_col]
z = df[z_col]

# Regression surface
model = LinearRegression()
X_fit = np.column_stack((x, y))
model.fit(X_fit, z)

# Grid for smooth surface
x_range = np.linspace(x.min(), x.max(), 30)
y_range = np.linspace(y.min(), y.max(), 30)
x_mesh, y_mesh = np.meshgrid(x_range, y_range)
z_pred = model.predict(np.column_stack((x_mesh.ravel(), y_mesh.ravel()))).reshape(x_mesh.shape)

# Plot surface and pins
surface = go.Surface(x=x_mesh, y=y_mesh, z=z_pred, colorscale="Viridis", opacity=0.6)

scatter = go.Scatter3d(
    x=x, y=y, z=z,
    mode='markers',
    marker=dict(
        size=5,
        color=z,
        colorscale='RdYlGn',
        cmin=df[z_col].min(),
        cmax=df[z_col].max(),
        colorbar=dict(title="Readiness")
    ),
    text=df["Base"],
    hovertemplate="Base: %{text}<br>" + x_col + ": %{x}<br>" + y_col + ": %{y}<br>Readiness: %{z}<extra></extra>"
)

fig = go.Figure(data=[surface, scatter])
fig.update_layout(
    scene=dict(
        xaxis_title=x_col,
        yaxis_title=y_col,
        zaxis_title=z_col,
    ),
    margin=dict(l=0, r=0, b=0, t=30),
    height=700
)

st.plotly_chart(fig, use_container_width=True)

st.subheader("📋 Interpretation")
st.markdown(f"""
- Readiness scores clearly decline as **Mission Complexity** and **Maintenance Burden** increase.
- The surface tilt and gradient show a strong inverse relationship.
- Bases with high complexity and maintenance show the **lowest readiness scores**.
""")
