# QEC Dashboard

표면 부호(surface code) **양자 오류정정(QEC) 시뮬레이터**를 FastAPI로 래핑한 웹 대시보드.
MLflow로 학습한 신경망 디코더(MLP)를 MWPM 베이스라인과 함께 서빙한다.
(전남대 AI융합대학 DevOps/MLOps 프로젝트)

- 배포: https://qec-dashboard.onrender.com

## 기능

- **시뮬레이션 / 디코딩**: surface code를 시뮬레이션하고 선택한 디코더로 LER 측정
- **디코더 선택**: MWPM(고전) 또는 MLflow registry의 학습된 NN 모델 (현재 geometry에 맞는 것만 표시)
- **단일 shot 시각화**: syndrome · data qubit 에러 · 보정을 격자로 표시
- **물리 큐빗 에러 통계**

## 실행

```bash
# MLflow 서버 (모델 레지스트리)
mlflow server --backend-store-uri sqlite:///mlruns.db --artifacts-destination ./mlartifacts --port 5000

# 학습 → 등록 → 승격
python -m ml.train --config ml/configs/mlp_d3.yaml
python -m ml.promote --name mlp_d3_r3 --to-alias champion --from-alias challenger

# 앱
uvicorn app.main:app --env-file .env --reload   # http://127.0.0.1:8000
```

`.env`는 `.env.example` 참고 (`MLFLOW_TRACKING_URI`, `GH_REPO`, `GH_TOKEN`).

## 구조

```
app/    FastAPI 앱 (엔드포인트 · 모델 로더 · 로깅/이슈)
ml/     학습/승격 스크립트 + config
tests/  pytest
static/ 대시보드 UI
```

시뮬레이터: [qec_sim](https://github.com/jlib245/qec) · 디코더 모델은 MLflow registry에서 로드.
