# tests/test_simulate_fuzz_real.py
import unittest
from hypothesis import given, assume, settings
from hypothesis import strategies as st

from app.simulate import run_simulation


class TestRunSimulationFuzzReal(unittest.TestCase):
    """
    run_simulation()에 대한 속성 기반 테스트 (Mock 없이 실제 qec_sim 호출).
    다양한 입력 조합에서 불변식이 항상 성립하는지 검증한다.
    """

    # ── 속성 1: LER은 항상 0.0 ~ 1.0 범위여야 한다 ────────────

    @given(
        distance=st.sampled_from([3, 5]),          # surface code는 홀수만 유효
        rounds=st.integers(min_value=1, max_value=5),
        p_gate=st.floats(min_value=0.0, max_value=0.3, allow_nan=False),
        p_meas=st.floats(min_value=0.0, max_value=0.3, allow_nan=False),
        shots=st.integers(min_value=1, max_value=50),
    )
    @settings(max_examples=1000, deadline=None)
    def test_ler_always_in_range(self, distance, rounds, p_gate, p_meas, shots):
        """어떤 유효한 파라미터 조합이더라도 LER은 [0, 1] 범위여야 한다"""
        result = run_simulation(
            distance=distance,
            rounds=rounds,
            p_gate=p_gate,
            p_meas=p_meas,
            p_leak=0.0,
            shots=shots,
        )
        self.assertEqual(0.0 <= result["ler"] <= 1.0, True)

    # ── 속성 2: 노이즈가 0이면 LER은 항상 0.0이어야 한다 ───────

    @given(
        distance=st.sampled_from([3, 5]),
        rounds=st.integers(min_value=1, max_value=5),
        shots=st.integers(min_value=1, max_value=50),
    )
    @settings(max_examples=1000, deadline=None)
    def test_ler_zero_when_no_noise(self, distance, rounds, shots):
        """p_gate=0, p_meas=0이면 LER은 항상 0.0이어야 한다"""
        result = run_simulation(
            distance=distance,
            rounds=rounds,
            p_gate=0.0,
            p_meas=0.0,
            p_leak=0.0,
            shots=shots,
        )
        self.assertEqual(result["ler"], 0.0)


if __name__ == "__main__":
    unittest.main()
