# app/visualize.py
from types import SimpleNamespace

import numpy as np

try:
    import stim
    from qec_sim.config.schema import CodeParams, NoiseParams
    from qec_sim.circuit.registry import build_circuit
    from qec_sim.decoders.mwpm import MWPMDecoder
    from qec_sim.decoders.lut import build_detector_lut, compute_lut_correction
except ImportError:
    # qec-sim/stim 미설치 환경(mock CI 등): 테스트에서 patch되거나, 호출 시 에러로 드러나야 함
    stim = None
    CodeParams = NoiseParams = SimpleNamespace
    build_circuit = MWPMDecoder = None
    build_detector_lut = compute_lut_correction = None


def _lut_correction(circuit, det_flips, obs_flips):  # pragma: no cover  (실제 qec_sim 필요 — integration/live로 검증)
    """LUT 기반 단일샷 보정 (MWPM 매칭 없이).

    - corrected qubits: detector→대표 에러의 data qubit 좌표 LUT를 발화 detector에 union
      (base.py의 explain_detector_error_model_errors와 동일 방식)
    - logical_error: observable LUT(build_detector_lut)로 logical 출력 보정 → 실제와 XOR
    """
    dem = circuit.detector_error_model(decompose_errors=True)

    # detector → 대표 에러가 flip하는 큐빗 좌표
    explained = circuit.explain_detector_error_model_errors(
        dem_filter=dem, reduce_to_one_representative_error=True
    )
    dem_errors = [i for i in dem.flattened() if i.type == "error"]
    det2qubits = {}
    for err_inst, expl in zip(dem_errors, explained):
        dets = [t.val for t in err_inst.targets_copy() if t.is_relative_detector_id()]
        qs = set()
        for loc in expl.circuit_error_locations:
            for gt in loc.flipped_pauli_product:
                c = gt.coords
                if c:
                    qs.add((float(c[0]), float(c[1])))
        for d in dets:
            det2qubits.setdefault(d, list(qs))

    coords = set()
    for d, fired in enumerate(det_flips):
        if fired:
            for q in det2qubits.get(d, []):
                coords.add(q)
    corrected_qubits = [{"x": x, "y": y} for x, y in coords]

    # logical: observable LUT base correction과 실제 observable XOR
    lut = build_detector_lut(circuit)
    lut_corr = compute_lut_correction(det_flips[np.newaxis, :], lut)[0]
    logical_error = bool(int(obs_flips[0]) ^ int(lut_corr[0]))

    return corrected_qubits, logical_error


def run_visualize(
    distance: int,
    rounds: int,
    p_gate: float,
    p_meas: float,
    decoder: str = "mwpm",
) -> dict:
    """
    단일 shot 시뮬레이션 + correction 시각화 정보 반환.
    FlipSimulator 단일 실행으로 syndrome/data qubit 에러/logical_error 동일 샘플.

    decoder="mwpm": 매칭으로 보정 큐빗 + logical.
    그 외: LUT 기반 보정 (NN은 logical만 출력하므로, detector→qubit LUT로 보정 큐빗을
    그리고 observable LUT로 logical 출력을 보정).
    """
    code_params = CodeParams(name="surface_code", distance=distance, rounds=rounds)
    noise_params = NoiseParams(p_gate=p_gate, p_meas=p_meas, p_corr=0.0)

    circuit = build_circuit(code_params.name, code_params, noise_params).build()

    flip_sim = stim.FlipSimulator(batch_size=1, disable_stabilizer_randomization=True)
    flip_sim.do(circuit)

    det_flips = np.array(flip_sim.get_detector_flips(bit_packed=False), dtype=int).flatten()
    obs_flips = np.array(flip_sim.get_observable_flips(bit_packed=False), dtype=int).flatten()
    flipped = flip_sim.peek_pauli_flips()[0]

    if decoder == "mwpm":
        error_model = circuit.detector_error_model(decompose_errors=True)
        mwpm = MWPMDecoder(error_model, circuit=circuit)
        correction_info = mwpm.decode_single_with_correction(det_flips)
        corrected_qubits = mwpm.get_corrected_qubits(correction_info["corrected_fault_ids"])
        logical_error = bool(obs_flips[0] ^ correction_info["logical_error"][0])
        correction_method = "mwpm"
    else:
        corrected_qubits, logical_error = _lut_correction(circuit, det_flips, obs_flips)
        correction_method = "lut"

    # data qubit vs ancilla는 좌표 parity로 분리 (rotated surface code 컨벤션):
    # - data qubit: (odd, odd)
    # - ancilla:    (even, even)  — X/Z 구분은 아래 (x+y) % 4 로직에서
    # detector 등록 여부에 의존하던 기존 방식은 rounds=1에서 X-ancilla가 detector_xy에
    # 없어서 data qubit으로 오분류되는 버그가 있었음.
    qubit_coords = circuit.get_final_qubit_coordinates()
    detector_coords = circuit.get_detector_coordinates()

    ancilla_xy_set = set()
    data_qubit_set = set()
    data_qubits = []
    for qubit_id, coord in qubit_coords.items():
        x, y = int(coord[0]), int(coord[1])
        if x % 2 == 0 and y % 2 == 0:
            ancilla_xy_set.add((coord[0], coord[1]))
        else:
            errored = int(flipped[qubit_id]) != 0 if qubit_id < len(flipped) else 0
            data_qubits.append({"x": coord[0], "y": coord[1], "errored": int(errored)})
            data_qubit_set.add((coord[0], coord[1]))

    all_ancilla_positions = [{"x": k[0], "y": k[1]} for k in sorted(ancilla_xy_set)]

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
        "correction_method": correction_method,
    }
