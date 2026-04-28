# app/simulate.py
from types import SimpleNamespace

import numpy as np

try:
    from qec_sim.config.schema import CodeParams, NoiseParams
    from qec_sim.circuit.registry import build_circuit
    from qec_sim.circuit.simulator import CircuitNoiseSimulator
    from qec_sim.decoders.mwpm import ErasureMWPM
except ImportError:
    # qec-sim 미설치 환경(mock CI 등): kwargs를 attr로 저장하는 stub만 두고,
    # 실제 호출되는 함수/클래스는 None — 테스트에서 patch되거나, 호출 시 에러로 드러나야 함
    CodeParams = NoiseParams = SimpleNamespace
    build_circuit = CircuitNoiseSimulator = ErasureMWPM = None


def run_simulation(
    distance: int,
    rounds: int,
    p_gate: float,
    p_meas: float,
    p_leak: float = 0.0,
    shots: int = 1000,
) -> dict:
    """
    Surface code 시뮬레이션 실행 후 LER 반환.

    Returns:
        {
            "ler": float
        }
    """
    code_params = CodeParams(name="surfacecode", distance=distance, rounds=rounds)
    noise_params = NoiseParams(p_gate=p_gate, p_meas=p_meas, p_corr=0.0, p_leak=p_leak)

    builder = build_circuit(code_params.name, code_params, noise_params)
    circuit = builder.build()

    simulator = CircuitNoiseSimulator(circuit, noise_params)
    data = simulator.generate_data(shots=shots)

    syndromes = data["syndromes"]      # (shots, num_detectors)
    observables = data["observables"]  # (shots, num_observables)
    erasures = data["erasures"]        # (shots, num_detectors)

    error_model = circuit.detector_error_model(decompose_errors=True)
    decoder = ErasureMWPM(error_model)
    predictions = decoder.decode_batch(syndromes, erasures)

    ler = float((predictions != observables).any(axis=1).mean())

    return {"ler": ler}
