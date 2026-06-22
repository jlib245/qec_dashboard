# app/simulate.py
import sys
from types import SimpleNamespace

import numpy as np


def _sanitize_prob(p: float) -> float:
    """Subnormal float은 detector error model의 edge weight overflow를 유발하므로 0.0으로 clamp."""
    if p < sys.float_info.min:
        return 0.0
    return p

try:
    from qec_sim.config.schema import CodeParams, NoiseParams
    from qec_sim.circuit.registry import build_circuit
    from qec_sim.circuit.simulator import CircuitNoiseSimulator
    from qec_sim.decoders.mwpm import MWPMDecoder
except ImportError:
    # qec-sim 미설치 환경(mock CI 등): kwargs를 attr로 저장하는 stub만 두고,
    # 실제 호출되는 함수/클래스는 None — 테스트에서 patch되거나, 호출 시 에러로 드러나야 함
    CodeParams = NoiseParams = SimpleNamespace
    build_circuit = CircuitNoiseSimulator = MWPMDecoder = None


def run_simulation(
    distance: int,
    rounds: int,
    p_gate: float,
    p_meas: float,
    shots: int = 1000,
) -> dict:
    """
    Surface code 시뮬레이션 실행 후 LER 반환.

    Returns:
        {
            "ler": float
        }
    """
    p_gate = _sanitize_prob(p_gate)
    p_meas = _sanitize_prob(p_meas)

    code_params = CodeParams(name="surface_code", distance=distance, rounds=rounds)
    noise_params = NoiseParams(p_gate=p_gate, p_meas=p_meas, p_corr=0.0)

    builder = build_circuit(code_params.name, code_params, noise_params)
    circuit = builder.build()

    simulator = CircuitNoiseSimulator(circuit, noise_params)
    data = simulator.generate_data(shots=shots)

    syndromes = data["syndromes"]      # (shots, num_detectors)
    observables = data["observables"]  # (shots, num_observables)

    error_model = circuit.detector_error_model(decompose_errors=True)
    decoder = MWPMDecoder(error_model)
    predictions = decoder.decode_batch(syndromes)

    ler = float((predictions != observables).any(axis=1).mean())

    return {"ler": ler}
