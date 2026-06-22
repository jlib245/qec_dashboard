# dashboard.py  (streamlit run dashboard.py)
from pathlib import Path

import pandas as pd
import streamlit as st

PREDICTION_LOG_PATH = Path("logs/predictions.csv")

st.set_page_config(page_title="QEC MLOps Dashboard", layout="wide")
st.title("QEC Dashboard 운영 모니터링")

if not PREDICTION_LOG_PATH.exists():
    st.info("아직 예측 로그가 없습니다. /decode 요청을 보내보세요.")
    st.stop()

pred_df = pd.read_csv(PREDICTION_LOG_PATH)
pred_df["ler"] = pd.to_numeric(pred_df["ler"], errors="coerce")

# -------- 운영 지표 --------
st.subheader("운영 지표")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Requests", len(pred_df))
col2.metric("Average LER", f"{pred_df['ler'].mean():.3%}")
col3.metric("High LER (>0.1)", int((pred_df["ler"] > 0.1).sum()))
col4.metric("Decoders Used", pred_df["decoder"].nunique())

# -------- LER 추이 --------
st.subheader("LER Trend")
trend_df = pred_df.reset_index()
st.line_chart(trend_df, x="index", y="ler")

# -------- decoder별 사용량 --------
st.subheader("Requests by Decoder")
st.bar_chart(pred_df["decoder"].value_counts())

# -------- geometry별 평균 LER --------
if {"distance", "rounds"}.issubset(pred_df.columns):
    st.subheader("Average LER by geometry")
    geo = pred_df.copy()
    geo["geometry"] = "d" + geo["distance"].astype(str) + "/r" + geo["rounds"].astype(str)
    st.bar_chart(geo.groupby("geometry")["ler"].mean())

# -------- 최근 요청 --------
st.subheader("Recent Requests")
st.dataframe(pred_df.tail(20), use_container_width=True)
