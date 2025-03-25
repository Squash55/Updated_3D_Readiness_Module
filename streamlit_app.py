
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
import numpy as np

st.set_page_config(page_title="Final 3D Readiness Chart (Artificial data)", layout="wide")

@st.cache_data
def load_data():
    return pd.read_csv("USAF_3D_Data.csv")

df = load_data()

st.title("📡 Final 3D Readiness Chart with Pins and Contours (Artificial data)")
st.markdown("This version includes surface contours, colored pins, and enhanced readability.")

x_col = "Mission Complexity"
y_col = "Maintenance Burden"
z_col = "Readiness Score"

x = df[x_col]
y = df[y_col]
z = df[z_col]

model = LinearRegression()
X = np.column_stack((x, y))
model.fit(X, z)
z_pred = model.predict(X)
r2 = model.score(X, z)

x_range = np.linspace(x.min(), x.max(), 30)
y_range = np.linspace(y.min(), y.max(), 30)
x_mesh, y_mesh = np.meshgrid(x_range, y_range)
z_mesh = model.predict(np.column_stack((x_mesh.ravel(), y_mesh.ravel()))).reshape(x_mesh.shape)

# Vertical pins
lines = []
for xi, yi, zi in zip(x, y, z):
    lines.append(go.Scatter3d(
        x=[xi, xi], y=[yi, yi], z=[0, zi],
        mode="lines",
        line=dict(color="white", width=2),
        showlegend=False
    ))

# Surface with contours and reduced opacity
surface = go.Surface(
    x=x_mesh, y=y_mesh, z=z_mesh,
    colorscale="Viridis",
    opacity=0.9,
    contours=dict(
        z=dict(
            show=True,
            usecolormap=True,
            highlightcolor="white",
            project_z=True
        )
    )
)

# Scatter dots
scatter = go.Scatter3d(
    x=x, y=y, z=z,
    mode="markers",
    marker=dict(
        size=6,
        color=z,
        colorscale="RdYlGn",
        colorbar=dict(title="Readiness"),
        cmin=df[z_col].min(),
        cmax=df[z_col].max()
    ),
    text=df["Base"],
    hovertemplate="Base: %{text}<br>" + x_col + ": %{x}<br>" + y_col + ": %{y}<br>Readiness: %{z}<extra></extra>"
)

fig = go.Figure(data=[surface, scatter] + lines)

fig.update_layout(
    scene=dict(
        xaxis_title=x_col,
        yaxis_title=y_col,
        zaxis_title=z_col,
        xaxis=dict(showgrid=True, gridcolor="white"),
        yaxis=dict(showgrid=True, gridcolor="white"),
        zaxis=dict(showgrid=True, gridcolor="white")
    ),
    margin=dict(l=0, r=0, b=0, t=30),
    height=750
)

st.plotly_chart(fig, use_container_width=True)

st.subheader("📊 Smart Interpretation Summary")
st.markdown(f'''
- This enhanced 3D chart includes surface contours and more visible slopes.
- Vertical pins and color-coded markers highlight each base’s readiness.
- The regression surface explains **{r2:.2f}** of readiness variance.
- Prioritize support for bases with high mission complexity and elevated maintenance.
''')
