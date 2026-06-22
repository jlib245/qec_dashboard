# app/prediction_logger.py
import csv
from datetime import datetime
from pathlib import Path

PREDICTION_LOG_PATH = Path("logs/predictions.csv")
PREDICTION_LOG_PATH.parent.mkdir(exist_ok=True)


def save_prediction_log(decoder, distance, rounds, p_gate, p_meas, shots, ler):
    """/decode 요청 1건을 CSV에 기록 (모니터링/대시보드용).

    (Render 파일시스템은 ephemeral — 영속화하려면 외부 DB/Sheets. 여기선 로컬 시연용 CSV.)
    """
    is_new = not PREDICTION_LOG_PATH.exists()
    with open(PREDICTION_LOG_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(
                ["time", "decoder", "distance", "rounds", "p_gate", "p_meas", "shots", "ler"]
            )
        writer.writerow([
            datetime.now().isoformat(timespec="seconds"),
            decoder, distance, rounds, p_gate, p_meas, shots,
            round(float(ler), 4),
        ])
