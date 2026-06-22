# app/google_sheet_logger.py
import os
import json
from datetime import datetime

import gspread
from google.oauth2.service_account import Credentials

SCOPE = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]

_spreadsheet = None


def get_spreadsheet():
    """서비스 계정으로 Google Sheet 열기 (Render/앱: JSON 문자열 env로 인증)."""
    global _spreadsheet
    if _spreadsheet is not None:
        return _spreadsheet

    sheet_name = os.getenv("GOOGLE_SHEET_NAME")
    service_account_json = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")
    if not sheet_name:
        raise RuntimeError("GOOGLE_SHEET_NAME is not set")
    if not service_account_json:
        raise RuntimeError("GOOGLE_SERVICE_ACCOUNT_JSON is not set")

    info = json.loads(service_account_json)
    credentials = Credentials.from_service_account_info(info, scopes=SCOPE)
    _spreadsheet = gspread.authorize(credentials).open(sheet_name)
    return _spreadsheet


def append_prediction_log(decoder, distance, rounds, p_gate, p_meas, shots, ler):
    """예측 로그 1줄을 'prediction_logs' 워크시트에 append (운영 모니터링 영속화)."""
    worksheet = get_spreadsheet().worksheet("prediction_logs")
    worksheet.append_row([
        datetime.now().isoformat(timespec="seconds"),
        decoder, distance, rounds, p_gate, p_meas, shots,
        round(float(ler), 4),
    ])
