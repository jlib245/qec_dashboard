# dashboard.py  (streamlit run dashboard.py)
import os
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()  # .env의 GOOGLE_SHEET_NAME / GOOGLE_SERVICE_ACCOUNT_FILE 읽기

PREDICTION_LOG_PATH = Path("logs/predictions.csv")
SCOPE = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]
GOOGLE_SHEET_NAME = os.getenv("GOOGLE_SHEET_NAME")
GOOGLE_SERVICE_ACCOUNT_FILE = os.getenv("GOOGLE_SERVICE_ACCOUNT_FILE", "google_key.json")


@st.cache_resource
def _get_spreadsheet():
    import gspread
    from google.oauth2.service_account import Credentials

    creds = Credentials.from_service_account_file(GOOGLE_SERVICE_ACCOUNT_FILE, scopes=SCOPE)
    return gspread.authorize(creds).open(GOOGLE_SHEET_NAME)


def load_predictions():
    """운영 로그 로드. Google Sheets(설정+키 있으면, 운영/Render 포함) 우선, 없으면 로컬 CSV."""
    if GOOGLE_SHEET_NAME and Path(GOOGLE_SERVICE_ACCOUNT_FILE).exists():
        ws = _get_spreadsheet().worksheet("prediction_logs")
        return pd.DataFrame(ws.get_all_records()), "Google Sheets"
    if PREDICTION_LOG_PATH.exists():
        return pd.read_csv(PREDICTION_LOG_PATH), "local CSV"
    return None, None


st.set_page_config(page_title="QEC MLOps Dashboard", layout="wide")
st.title("QEC Dashboard 운영 모니터링")

pred_df, source = load_predictions()
if pred_df is None or len(pred_df) == 0:
    st.info("아직 예측 로그가 없습니다. /decode 요청을 보내보세요.")
    st.stop()

st.caption(f"source: {source}")
pred_df["ler"] = pd.to_numeric(pred_df["ler"], errors="coerce")


def _model_label(row):
    """모델 + version 표기 (alias 대신 버전). 예: mlp_d3_r3 v3 / mwpm."""
    v = row.get("version")
    if pd.notna(v) and str(v).strip() not in ("", "nan"):
        return f"{row['decoder']} v{int(float(v))}"
    return str(row["decoder"])


pred_df["model"] = pred_df.apply(_model_label, axis=1)

# -------- 운영 지표 --------
st.subheader("운영 지표")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Requests", len(pred_df))
col2.metric("Average LER", f"{pred_df['ler'].mean():.3%}")
col3.metric("High LER (>0.1)", int((pred_df["ler"] > 0.1).sum()))
col4.metric("Models Used", pred_df["model"].nunique())

# -------- LER 추이 --------
st.subheader("LER Trend")
st.line_chart(pred_df.reset_index(), x="index", y="ler")

# -------- 모델(버전)별 사용량 --------
st.subheader("Requests by Model (version)")
st.bar_chart(pred_df["model"].value_counts())

# -------- geometry별 평균 LER --------
if {"distance", "rounds"}.issubset(pred_df.columns):
    st.subheader("Average LER by geometry")
    geo = pred_df.copy()
    geo["geometry"] = "d" + geo["distance"].astype(str) + "/r" + geo["rounds"].astype(str)
    st.bar_chart(geo.groupby("geometry")["ler"].mean())

# -------- 최근 요청 --------
st.subheader("Recent Requests")
st.dataframe(pred_df.tail(20), use_container_width=True)
