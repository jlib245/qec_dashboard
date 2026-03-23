# app/simulate.py
import numpy as np
from qec_sim.config.schema import CodeParams, NoiseParams
from qec_sim.circuit.registry import build_circuit
from qec_sim.circuit.simulator import CircuitNoiseSimulator
from qec_sim.decoders.mwpm import ErasureMWPM
from qec_sim.data.preprocessors import SpatialGridPreprocessor


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
            "ler": float,
            "syndrome_grid": list,   # 첫 번째 shot의 syndrome (rounds x grid_h x grid_w)
            "erasure_grid": list,    # 첫 번째 shot의 erasure (rounds x grid_h x grid_w)
            "prediction": list,      # 첫 번째 shot의 논리 에러 예측 결과
            "grid_h": int,
            "grid_w": int,
            "detector_coords": dict, # detector 좌표 {id: [x, y, t]}
        }
    """
    code_params = CodeParams(name="surface_code", distance=distance, rounds=rounds)
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

    # LER: 예측이 틀린 shot 비율
    ler = float((predictions != observables).any(axis=1).mean())

    # SpatialGridPreprocessor로 detector 좌표 → 2D grid 매핑
    detector_coords = circuit.get_detector_coordinates()
    preprocessor = SpatialGridPreprocessor(
        detector_coords=detector_coords,
        num_detectors=circuit.num_detectors,
        use_erasures=True,
    )

    # 첫 번째 shot을 (rounds x grid_h x grid_w) grid로 변환
    syn_first = syndromes[0]   # (num_detectors,)
    era_first = erasures[0]    # (num_detectors,)

    syndrome_grid = np.zeros((rounds, preprocessor.grid_h, preprocessor.grid_w), dtype=int)
    erasure_grid = np.zeros((rounds, preprocessor.grid_h, preprocessor.grid_w), dtype=int)

    for det_idx, c, h, w in zip(
        preprocessor.det_idx.tolist(),
        preprocessor.c_idx.tolist(),
        preprocessor.h_idx.tolist(),
        preprocessor.w_idx.tolist(),
    ):
        t = c  # c_idx == round index
        if t < rounds:
            syndrome_grid[t, h, w] = int(syn_first[det_idx])
            erasure_grid[t, h, w] = int(era_first[det_idx])

    return {
        "ler": ler,
        "syndrome_grid": syndrome_grid.tolist(),
        "erasure_grid": erasure_grid.tolist(),
        "prediction": predictions[0].astype(int).tolist(),
        "grid_h": preprocessor.grid_h,
        "grid_w": preprocessor.grid_w,
        "detector_coords": {str(k): list(v) for k, v in detector_coords.items()},
    }
