# app/visualize.py
import stim
import numpy as np
from qec_sim.config.schema import CodeParams, NoiseParams
from qec_sim.circuit.registry import build_circuit
from qec_sim.decoders.mwpm import ErasureMWPM


def run_visualize(
    distance: int,
    rounds: int,
    p_gate: float,
    p_meas: float,
    p_leak: float = 0.0,
) -> dict:
    """
    단일 shot 시뮬레이션 + MWPM correction 시각화 정보 반환.
    FlipSimulator 단일 실행으로 syndrome/data qubit 에러/logical_error 동일 샘플.
    (p_meas가 최종 data qubit 측정에도 X_ERROR를 넣는 점은 감안)
    """
    code_params = CodeParams(name="surface_code", distance=distance, rounds=rounds)
    noise_params = NoiseParams(p_gate=p_gate, p_meas=p_meas, p_corr=0.0, p_leak=p_leak)

    circuit = build_circuit(code_params.name, code_params, noise_params).build()

    flip_sim = stim.FlipSimulator(batch_size=1, disable_stabilizer_randomization=True)
    flip_sim.do(circuit)

    det_flips = np.array(flip_sim.get_detector_flips(bit_packed=False), dtype=int).flatten()
    obs_flips = np.array(flip_sim.get_observable_flips(bit_packed=False), dtype=int).flatten()
    flipped = flip_sim.peek_pauli_flips()[0]

    # MWPM decoder
    error_model = circuit.detector_error_model(decompose_errors=True)
    decoder = ErasureMWPM(error_model, circuit=circuit)
    correction_info = decoder.decode_single_with_correction(det_flips)
    corrected_qubits = decoder.get_corrected_qubits(correction_info["corrected_fault_ids"])

    logical_error = bool(obs_flips[0] ^ correction_info["logical_error"][0])

    # data qubit 좌표 + errored
    qubit_coords = circuit.get_final_qubit_coordinates()
    detector_coords = circuit.get_detector_coordinates()
    detector_xy = {(v[0], v[1]) for v in detector_coords.values()}

    data_qubit_set = set()
    data_qubits = []
    for qubit_id, coord in qubit_coords.items():
        if (coord[0], coord[1]) not in detector_xy:
            errored = int(flipped[qubit_id]) != 0 if qubit_id < len(flipped) else 0
            data_qubits.append({"x": coord[0], "y": coord[1], "errored": int(errored)})
            data_qubit_set.add((coord[0], coord[1]))

    # round별 ancilla 상태
    # X/Z 구분: (x+y) % 4 == 2 → X stabilizer, == 0 → Z stabilizer
    unique_ancilla_xy = {}
    for coord in detector_coords.values():
        unique_ancilla_xy[(coord[0], coord[1])] = True

    # X ancilla: qubit_coords에 있지만 detector_xy에 없는 (even, even) 위치
    for qubit_id, coord in qubit_coords.items():
        xy = (coord[0], coord[1])
        if xy not in detector_xy and int(coord[0]) % 2 == 0 and int(coord[1]) % 2 == 0:
            unique_ancilla_xy[xy] = True

    all_ancilla_positions = [{"x": k[0], "y": k[1]} for k in sorted(unique_ancilla_xy.keys())]

    # stabilizer edges: 각 ancilla (ax, ay)의 대각선 4방향 data qubit 연결
    edges = []
    for pos in all_ancilla_positions:
        ax, ay = pos["x"], pos["y"]
        for dx, dy in [(-1, -1), (1, -1), (-1, 1), (1, 1)]:
            nb = (ax + dx, ay + dy)
            if nb in data_qubit_set:
                edges.append({"x0": ax, "y0": ay, "x1": nb[0], "y1": nb[1]})

    ancillas_by_round = []
    for t in range(rounds):
        round_map = {}
        for det_id, coord in detector_coords.items():
            if len(coord) > 2 and int(coord[2]) == t:
                round_map[(coord[0], coord[1])] = {
                    "fired": int(det_flips[int(det_id)]) if int(det_id) < len(det_flips) else 0,
                    "erased": 0,
                }
        round_ancillas = []
        for pos in all_ancilla_positions:
            key = (pos["x"], pos["y"])
            state = round_map.get(key, {"fired": 0, "erased": 0})
            stabilizer_type = "X" if int(pos["x"] + pos["y"]) % 4 == 2 else "Z"
            round_ancillas.append({
                "x": pos["x"], "y": pos["y"],
                "fired": state["fired"],
                "erased": state["erased"],
                "active": 1 if key in round_map else 0,
                "stabilizer_type": stabilizer_type,
            })
        ancillas_by_round.append(round_ancillas)

    return {
        "logical_error": logical_error,
        "data_qubits": data_qubits,
        "ancillas": ancillas_by_round[0],
        "ancillas_by_round": ancillas_by_round,
        "corrected_qubits": corrected_qubits,
        "edges": edges,
    }
