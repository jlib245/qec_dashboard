# app/simulate.py
from qec_sim.config.schema import CodeParams, NoiseParams
from qec_sim.circuit.registry import build_circuit
from qec_sim.circuit.simulator import CircuitNoiseSimulator
from qec_sim.decoders.mwpm import ErasureMWPM


def run_simulation(
    distance: int,
    rounds: int,
    p_gate: float,
    p_meas: float,
    p_leak: float = 0.0,
    shots: int = 1000,
) -> dict:
    """
    Surface code 시뮬레이션 실행 후 결과 반환.

    Returns:
        {
            "ler": float,                  # Logical Error Rate
            "syndrome_sample": list[list], # 첫 번째 shot의 syndrome grid (rounds x detectors_per_round)
            "erasure_sample": list[list],  # 첫 번째 shot의 erasure grid
        }
    """
    code_params = CodeParams(name="surface_code", distance=distance, rounds=rounds)
    noise_params = NoiseParams(p_gate=p_gate, p_meas=p_meas, p_corr=0.0, p_leak=p_leak)

    builder = build_circuit(code_params.name, code_params, noise_params)
    circuit = builder.build()

    simulator = CircuitNoiseSimulator(circuit, noise_params)
    data = simulator.generate_data(shots=shots)

    syndromes = data["syndromes"]   # (shots, num_detectors)
    observables = data["observables"]  # (shots, num_observables)
    erasures = data["erasures"]     # (shots, num_detectors)

    error_model = circuit.detector_error_model(decompose_errors=True)
    decoder = ErasureMWPM(error_model)
    predictions = decoder.decode_batch(syndromes, erasures)

    # LER: 예측이 틀린 shot 비율
    ler = float((predictions != observables).any(axis=1).mean())

    # 시각화용: 첫 번째 shot의 syndrome을 rounds x detectors_per_round 로 reshape
    detectors_per_round = simulator.num_detectors // rounds
    syndrome_2d = syndromes[0].reshape(rounds, detectors_per_round).astype(int).tolist()
    erasure_2d = erasures[0].reshape(rounds, detectors_per_round).astype(int).tolist()

    return {
        "ler": ler,
        "syndrome_sample": syndrome_2d,
        "erasure_sample": erasure_2d,
    }
