# app/stats.py
from types import SimpleNamespace

import numpy as np

try:
    import stim
    from qec_sim.config.schema import CodeParams, NoiseParams
    from qec_sim.circuit.registry import build_circuit
except ImportError:
    # qec-sim/stim 미설치 환경(mock CI 등): 테스트에서 patch되거나, 호출 시 에러로 드러나야 함
    stim = None
    CodeParams = NoiseParams = SimpleNamespace
    build_circuit = None


def run_stats(
    distance: int,
    rounds: int,
    p_gate: float,
    p_meas: float,
    shots: int = 1000,
) -> dict:
    """
    FlipSimulator batch로 실제 data qubit 에러 통계 반환.

    Returns:
        {
            "avg_errors": float,      # shot당 평균 에러난 data qubit 수
            "avg_error_rate": float,  # avg_errors / n_data_qubits
            "n_data_qubits": int,
        }
    """
    code_params = CodeParams(name="surface_code", distance=distance, rounds=rounds)
    noise_params = NoiseParams(p_gate=p_gate, p_meas=p_meas, p_corr=0.0)

    circuit = build_circuit(code_params.name, code_params, noise_params).build()

    qubit_coords = circuit.get_final_qubit_coordinates()
    detector_xy = {(v[0], v[1]) for v in circuit.get_detector_coordinates().values()}
    data_qubit_ids = [
        qid for qid, c in qubit_coords.items()
        if (c[0], c[1]) not in detector_xy
    ]
    n_data = len(data_qubit_ids)

    flip_sim = stim.FlipSimulator(batch_size=shots, disable_stabilizer_randomization=True)
    flip_sim.do(circuit)
    flips = flip_sim.peek_pauli_flips()

    avg_errors = float(np.mean([
        sum(1 for qid in data_qubit_ids if qid < len(s) and int(s[qid]) != 0)
        for s in flips
    ]))

    return {
        "avg_errors": round(avg_errors, 4),
        "avg_error_rate": round(avg_errors / n_data, 4),
        "n_data_qubits": n_data,
    }
