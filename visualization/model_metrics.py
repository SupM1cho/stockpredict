# visualization/model_metrics.py
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

def plot_loss_curve(history):
    """
    Plot loss curve dari training history (dict dengan epoch & loss).
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=history["train_loss"], mode="lines", name="Train Loss"))
    if "val_loss" in history:
        fig.add_trace(go.Scatter(y=history["val_loss"], mode="lines", name="Validation Loss"))
    fig.update_layout(title="Training Loss Curve", xaxis_title="Epoch", yaxis_title="Loss")
    st.plotly_chart(fig, use_container_width=True)


def plot_feature_importance(features, importances):
    """
    Bar chart untuk feature importance.
    """
    fig = px.bar(
        x=features,
        y=importances,
        labels={"x": "Feature", "y": "Importance"},
        title="Feature Importance"
    )
    st.plotly_chart(fig, use_container_width=True)


def plot_residuals(y_true, y_pred):
    """
    Residual plot untuk analisis error.
    """
    residuals = [yt - yp for yt, yp in zip(y_true, y_pred)]
    fig = px.scatter(x=y_pred, y=residuals, labels={"x": "Predicted", "y": "Residuals"})
    fig.update_layout(title="Residual Plot")
    st.plotly_chart(fig, use_container_width=True)
