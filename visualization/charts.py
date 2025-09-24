# visualization/charts.py
import streamlit as st
import plotly.graph_objects as go

def plot_candlestick(df, title="Candlestick Chart"):
    """
    Membuat candlestick chart interaktif dari dataframe.
    DataFrame harus punya kolom: Date, Open, High, Low, Close
    """
    fig = go.Figure(data=[go.Candlestick(
        x=df["Date"],
        open=df["Open"],
        high=df["High"],
        low=df["Low"],
        close=df["Close"]
    )])
    fig.update_layout(title=title, xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)


def plot_line(df, columns=["Close"], title="Stock Price Line Chart"):
    """
    Line chart untuk harga saham (Close, MA20, MA50, dll).
    """
    fig = go.Figure()
    for col in columns:
        if col in df.columns:
            fig.add_trace(go.Scatter(x=df["Date"], y=df[col], mode="lines", name=col))
    fig.update_layout(title=title)
    st.plotly_chart(fig, use_container_width=True)


def plot_volume(df, title="Trading Volume"):
    """
    Bar chart untuk volume trading.
    """
    fig = go.Figure(data=[go.Bar(x=df["Date"], y=df["Volume"], name="Volume")])
    fig.update_layout(title=title)
    st.plotly_chart(fig, use_container_width=True)
