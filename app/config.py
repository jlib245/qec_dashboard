# app/config.py
import os

# 디코더 모드 스위치 (SpamCheck의 MODEL_MODE rules/ml에 대응).
#   "mwpm"   — MWPMDecoder (고전 베이스라인)
#   "neural" — 학습된 NN 디코더 (feature/mlflow-decoder에서 연결)
MODEL_MODE = os.getenv("MODEL_MODE", "mwpm")

# MLflow (feature/mlflow-decoder에서 사용).
# ngrok 무료 도메인은 매번 바뀌므로 하드코딩 금지 — 항상 env로 주입.
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "")
MODEL_URI = os.getenv("MODEL_URI", "models:/mlp_d3_r3@champion")
# registry에는 core_model만 등록되므로, 서빙 시 wrapper 재조립용 config가 필요.
BUNDLED_CONFIG = os.getenv("BUNDLED_CONFIG", "ml/configs/mlp_d3.yaml")

# NN 모델은 특정 geometry 전용 (syndrome 길이 = num_detectors가 distance/rounds에 묶임).
# /decode는 이 고정 geometry에서 동작 — 학습 config와 반드시 일치해야 함.
FIXED_DISTANCE = int(os.getenv("FIXED_DISTANCE", "3"))
FIXED_ROUNDS = int(os.getenv("FIXED_ROUNDS", "3"))
