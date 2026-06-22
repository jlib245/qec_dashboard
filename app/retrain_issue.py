# app/retrain_issue.py
import logging
from datetime import datetime

from app.config import LER_DRIFT_THRESHOLD, DRIFT_LIMIT
from app.issue import create_github_issue

logger = logging.getLogger(__name__)

# 서버 실행 동안만 유지 (간단하게) — 수업 retrain_issue와 동일 구조
_state = {
    "high_ler_count": 0,
    "samples": [],
    "issue_created": False,
}


def update_issue_state(decoder, distance, rounds, ler, threshold):
    """drift(열화) 의심 시 GitHub Issue 생성.

    서빙 LER이 threshold를 초과한 요청이 누적되면(= 현재 입력에 모델이 잘 안 맞음)
    DRIFT_LIMIT회에서 한 번 이슈를 만든다. (수업: confidence 낮아짐 → 여기선 LER 높아짐)
    """
    if ler > threshold:
        _state["high_ler_count"] += 1
        _state["samples"].append({
            "decoder": decoder,
            "distance": distance,
            "rounds": rounds,
            "ler": round(float(ler), 4),
            "time": datetime.now().isoformat(timespec="seconds"),
        })

        if _state["high_ler_count"] >= DRIFT_LIMIT and not _state["issue_created"]:
            create_drift_issue()
            _state["issue_created"] = True

    return _state


def create_drift_issue():
    samples = _state["samples"][-DRIFT_LIMIT:]
    title = "[MLOps] Drift suspected (high-LER accumulation)"
    body = f"""
## Drift Detection Report
High-LER predictions accumulated.
- count: {_state["high_ler_count"]}
- threshold: {LER_DRIFT_THRESHOLD}
- limit: {DRIFT_LIMIT}

## Recent Samples
"""
    for s in samples:
        body += f"- LER={s['ler']} | decoder={s['decoder']} d={s['distance']}/r={s['rounds']}\n"
    body += """
## Action
- Please review input distribution (noise/geometry)
- Decide whether retraining is needed
"""
    create_github_issue(title, body, logger)
