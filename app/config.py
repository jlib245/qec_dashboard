# app/config.py
import os

# MLflow Tracking 서버 위치 (로컬: http://127.0.0.1:5000, 배포: ngrok 등으로 노출한 주소).
# ngrok 무료 도메인은 매번 바뀌므로 하드코딩 금지 — 항상 env로 주입.
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")

# (폴백) wrapper 재조립용 config. 보통은 모델 run의 config.yaml을 자동 다운로드하므로 미사용.
BUNDLED_CONFIG = os.getenv("BUNDLED_CONFIG", "ml/configs/mlp_d3.yaml")

# /decode·/visualize에서 decoder를 mwpm으로 쓸 때의 기본 geometry.
FIXED_DISTANCE = int(os.getenv("FIXED_DISTANCE", "3"))
FIXED_ROUNDS = int(os.getenv("FIXED_ROUNDS", "3"))
