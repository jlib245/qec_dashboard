# app/decode.py
from types import SimpleNamespace

import numpy as np

from app.simulate import _sanitize_prob

try:
    from qec_sim.config.schema import CodeParams, NoiseParams
    from qec_sim.circuit.registry import build_circuit
    from qec_sim.circuit.simulator import CircuitNoiseSimulator
    from qec_sim.decoders.mwpm import MWPMDecoder
except ImportError:
    # qec-sim 미설치 환경(mock CI 등): 테스트에서 patch되거나, 호출 시 에러로 드러나야 함
    CodeParams = NoiseParams = SimpleNamespace
    build_circuit = CircuitNoiseSimulator = MWPMDecoder = None


def _generate(distance: int, rounds: int, p_gate: float, p_meas: float, shots: int):
    """주어진 geometry/noise로 회로 생성 + shots만큼 샘플링."""
    code_params = CodeParams(name="surface_code", distance=distance, rounds=rounds)
    noise_params = NoiseParams(
        p_gate=_sanitize_prob(p_gate), p_meas=_sanitize_prob(p_meas), p_corr=0.0
    )
    circuit = build_circuit(code_params.name, code_params, noise_params).build()
    data = CircuitNoiseSimulator(circuit, noise_params).generate_data(shots=shots)
    return circuit, data["syndromes"], data["observables"]


def run_decode(
    decoder: str,
    p_gate: float,
    p_meas: float,
    shots: int = 1000,
    distance: int = 3,
    rounds: int = 3,
) -> dict:
    """선택한 디코더로 LER 계산.

    decoder == "mwpm":  입력 distance/rounds에서 MWPM (어떤 geometry든 가능).
    decoder == model_uri ("models:/name@alias"):  registry에서 그 neural 모델 로드.
        geometry는 모델 자신이 결정(입력 distance/rounds 무시) — NN은 학습 geometry 전용.

    Returns:
        {"decoder", "distance", "rounds", "ler", ("run_id" — neural만)}
    """
    if decoder == "mwpm":
        circuit, syndromes, observables = _generate(distance, rounds, p_gate, p_meas, shots)
        error_model = circuit.detector_error_model(decompose_errors=True)
        preds = MWPMDecoder(error_model).decode_batch(syndromes)
        ler = float((preds != observables).any(axis=1).mean())
        return {"decoder": "mwpm", "distance": distance, "rounds": rounds, "ler": ler}

    # neural: 모델이 자기 geometry를 결정 (registry의 config.yaml 기반)
    from app import model_loader
    rec = model_loader.load(decoder)
    d, r = rec["distance"], rec["rounds"]
    _, syndromes, observables = _generate(d, r, p_gate, p_meas, shots)
    preds = rec["decoder"].decode_batch(syndromes, batch_size=4096)
    ler = float((preds != observables).any(axis=1).mean())
    return {"decoder": decoder, "distance": d, "rounds": r, "ler": ler, "run_id": rec["run_id"]}
