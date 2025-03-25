
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
import numpy as np

st.set_page_config(page_title="Enhanced 3D Readiness Chart with Pins (Artificial data)", layout="wide")

@st.cache_data
def load_data():
    return pd.read_csv("USAF_3D_Data.csv")

df = load_data()

st.title("📡 Enhanced 3D Readiness Chart with Pins (Artificial data)")
st.markdown("This interactive chart includes colored pins from each data point to the base for better height perception.")

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

# Create vertical lines ("pins")
lines = []
for xi, yi, zi in zip(x, y, z):
    lines.append(go.Scatter3d(
        x=[xi, xi], y=[yi, yi], z=[0, zi],
        mode="lines",
        line=dict(color="white", width=2),
        showlegend=False
    ))

# Create surface and scatter points
surface = go.Surface(x=x_mesh, y=y_mesh, z=z_mesh, colorscale="Viridis", opacity=0.6)

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
    hovertemplate="Base: %{text}<br>Mission Complexity: %{x}<br>Maintenance Burden: %{y}<br>Readiness: %{z}<extra></extra>"
)

fig = go.Figure(data=[surface, scatter] + lines)

# Update layout with white grid lines
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
- This 3D surface plot shows a **strong inverse relationship** between readiness and both mission complexity and maintenance burden.
- The regression surface explains **{r2:.2f}** of the variance in readiness (R² score).
- Colored pins enhance height visibility for each base.
- White grid lines provide better contrast for visual orientation.
- Prioritize interventions at bases with high mission complexity and unresolved maintenance burdens.
''')
