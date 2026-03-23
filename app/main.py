# app/main.py
from fastapi import FastAPI, Body
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from app.simulate import run_simulation

app = FastAPI(title="QEC Dashboard")

app.mount("/static", StaticFiles(directory="static"), name="static")


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
                    "p_leak": 0.0,
                    "shots": 1000,
                },
            }
        },
    )
):
    result = run_simulation(
        distance=payload["distance"],
        rounds=payload["rounds"],
        p_gate=payload["p_gate"],
        p_meas=payload["p_meas"],
        p_leak=payload.get("p_leak", 0.0),
        shots=payload.get("shots", 1000),
    )
    return result
