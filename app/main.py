# app/main.py
import traceback
from typing import Callable

from fastapi import FastAPI, Body
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.simulate import run_simulation
from app.visualize import run_visualize
from app.stats import run_stats
from app.decode import run_decode
from app.logging_config import setup_logging, get_logger
from app.issue import create_github_issue

setup_logging()
logger = get_logger("qec_dashboard")

app = FastAPI(title="QEC Dashboard")

app.mount("/static", StaticFiles(directory="static"), name="static")


def _handle(endpoint: str, payload: dict, run: Callable):
    """모든 엔드포인트 공통 처리: CALL/OK/FAIL 로깅 + 실패 시 GitHub Issue 자동 생성.

    - 진입: CALL 로그 (endpoint + params)
    - 성공: OK 로그 후 결과 그대로 반환
    - 실패: FAIL 로그(스택트레이스) + create_github_issue + 500 응답
    """
    logger.info(f"CALL {endpoint} | params={payload}")
    try:
        result = run()
        logger.info(f"OK {endpoint}")
        return result
    except Exception as e:
        # 현재 예외의 traceback을 자동으로 찍는다 (파일/라인 포함)
        logger.exception(
            f"FAIL {endpoint} | params={payload} | error={type(e).__name__}: {e}"
        )
        tb = traceback.format_exc()
        title = f"[Prod Error] {endpoint} failed: {type(e).__name__}"
        body = (
            f"## Summary\n"
            f"- endpoint: {endpoint}\n"
            f"- params: `{payload}`\n\n"
            f"## Exception\n"
            f"- type: {type(e).__name__}\n"
            f"- message: {str(e)}\n\n"
            f"## Traceback (line info)\n"
            f"```text\n{tb}\n```"
        )
        create_github_issue(title, body, logger)
        return JSONResponse(
            status_code=500,
            content={"error": "Internal Server Error", "endpoint": endpoint},
        )


@app.get("/", response_class=HTMLResponse)
def home():
    with open("static/index.html", encoding="utf-8") as f:
        return f.read()


@app.post("/simulate")
async def simulate(
    payload: dict = Body(
        ...,
        examples={
            "default": {
                "summary": "기본 예시",
                "value": {
                    "distance": 3,
                    "rounds": 3,
                    "p_gate": 0.01,
                    "p_meas": 0.01,
                    "shots": 1000,
                },
            }
        },
    )
):
    return _handle(
        "/simulate",
        payload,
        lambda: run_simulation(
            distance=payload["distance"],
            rounds=payload["rounds"],
            p_gate=payload["p_gate"],
            p_meas=payload["p_meas"],
            shots=payload.get("shots", 1000),
        ),
    )


@app.post("/visualize")
async def visualize(
    payload: dict = Body(
        ...,
        examples={
            "default": {
                "summary": "기본 예시",
                "value": {
                    "distance": 3,
                    "rounds": 3,
                    "p_gate": 0.01,
                    "p_meas": 0.01,
                },
            }
        },
    )
):
    return _handle(
        "/visualize",
        payload,
        lambda: run_visualize(
            distance=payload["distance"],
            rounds=payload["rounds"],
            p_gate=payload["p_gate"],
            p_meas=payload["p_meas"],
        ),
    )


@app.post("/stats")
async def stats(
    payload: dict = Body(
        ...,
        examples={
            "default": {
                "summary": "기본 예시",
                "value": {
                    "distance": 3,
                    "rounds": 3,
                    "p_gate": 0.01,
                    "p_meas": 0.01,
                    "shots": 1000,
                },
            }
        },
    )
):
    return _handle(
        "/stats",
        payload,
        lambda: run_stats(
            distance=payload["distance"],
            rounds=payload["rounds"],
            p_gate=payload["p_gate"],
            p_meas=payload["p_meas"],
            shots=payload.get("shots", 1000),
        ),
    )


@app.post("/decode")
async def decode(
    payload: dict = Body(
        ...,
        examples={
            "default": {
                "summary": "기본 예시 (geometry는 서버 고정 d=3/r=3)",
                "value": {
                    "p_gate": 0.01,
                    "p_meas": 0.01,
                    "shots": 1000,
                },
            }
        },
    )
):
    return _handle(
        "/decode",
        payload,
        lambda: run_decode(
            p_gate=payload["p_gate"],
            p_meas=payload["p_meas"],
            shots=payload.get("shots", 1000),
        ),
    )
