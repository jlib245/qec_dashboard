# app/decode.py
from types import SimpleNamespace

import numpy as np

from app import config
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


def run_decode(p_gate: float, p_meas: float, shots: int = 1000) -> dict:
    """고정 geometry(config.FIXED_DISTANCE/ROUNDS)에서 MODEL_MODE 디코더로 LER 계산.

    동일 syndrome에 대해 MODEL_MODE에 따라 디코더만 바꿔 LER을 잰다 (mwpm vs neural 비교용).

    Returns:
        {"mode": str, "distance": int, "rounds": int, "ler": float}
    """
    distance = config.FIXED_DISTANCE
    rounds = config.FIXED_ROUNDS

    p_gate = _sanitize_prob(p_gate)
    p_meas = _sanitize_prob(p_meas)

    code_params = CodeParams(name="surface_code", distance=distance, rounds=rounds)
    noise_params = NoiseParams(p_gate=p_gate, p_meas=p_meas, p_corr=0.0)
    circuit = build_circuit(code_params.name, code_params, noise_params).build()

    simulator = CircuitNoiseSimulator(circuit, noise_params)
    data = simulator.generate_data(shots=shots)
    syndromes = data["syndromes"]
    observables = data["observables"]

    mode = config.MODEL_MODE
    if mode == "mwpm":
        error_model = circuit.detector_error_model(decompose_errors=True)
        predictions = MWPMDecoder(error_model).decode_batch(syndromes)
    elif mode == "neural":
        # mlflow는 neural 경로에서만 필요 → lazy import (mwpm-only 환경은 영향 없음).
        from app import model_loader
        predictions = model_loader.get_decoder().decode_batch(syndromes, batch_size=4096)
    else:
        raise ValueError(f"알 수 없는 MODEL_MODE: '{mode}' (mwpm | neural)")

    ler = float((predictions != observables).any(axis=1).mean())
    return {"mode": mode, "distance": distance, "rounds": rounds, "ler": ler}


def run_compare(p_gate: float, p_meas: float, shots: int = 1000) -> dict:
    """동일 syndrome에 대해 MWPM과 Neural 디코더의 LER을 함께 계산해 반환 (비교뷰용).

    MWPM은 항상 계산. Neural은 best-effort — 모델/MLflow 설정이 없으면
    neural_ler=None + neural_error로 알리고 MWPM 결과만 보여준다(서비스는 안 죽음).

    Returns:
        {"distance", "rounds", "shots", "mwpm_ler", "neural_ler"|None, "neural_error"|None}
    """
    distance = config.FIXED_DISTANCE
    rounds = config.FIXED_ROUNDS

    p_gate = _sanitize_prob(p_gate)
    p_meas = _sanitize_prob(p_meas)

    code_params = CodeParams(name="surface_code", distance=distance, rounds=rounds)
    noise_params = NoiseParams(p_gate=p_gate, p_meas=p_meas, p_corr=0.0)
    circuit = build_circuit(code_params.name, code_params, noise_params).build()

    data = CircuitNoiseSimulator(circuit, noise_params).generate_data(shots=shots)
    syndromes, observables = data["syndromes"], data["observables"]

    # MWPM (항상)
    error_model = circuit.detector_error_model(decompose_errors=True)
    mwpm_pred = MWPMDecoder(error_model).decode_batch(syndromes)
    mwpm_ler = float((mwpm_pred != observables).any(axis=1).mean())

    # Neural (best-effort — 모델 미설정 시 graceful degrade)
    neural_ler = None
    neural_error = None
    try:
        from app import model_loader
        nn_pred = model_loader.get_decoder().decode_batch(syndromes, batch_size=4096)
        neural_ler = float((nn_pred != observables).any(axis=1).mean())
    except Exception as e:
        neural_error = f"{type(e).__name__}: {e}"

    return {
        "distance": distance,
        "rounds": rounds,
        "shots": shots,
        "mwpm_ler": mwpm_ler,
        "neural_ler": neural_ler,
        "neural_error": neural_error,
    }
