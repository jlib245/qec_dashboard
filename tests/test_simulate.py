# tests/test_simulate.py
import unittest
from app.simulate import run_simulation


class TestRunSimulation(unittest.TestCase):
    """run_simulation() 단위 테스트"""

    def setUp(self):
        # 모든 테스트에서 공통으로 쓸 기본 파라미터
        self.base_params = dict(
            distance=3,
            rounds=3,
            p_gate=0.01,
            p_meas=0.01,
            p_leak=0.0,
            shots=200,
        )

    # ── 정상 동작 ──────────────────────────────────────────────

    def test_returns_ler_key(self):
        """반환값에 'ler' 키가 있어야 한다"""
        result = run_simulation(**self.base_params)
        self.assertEqual("ler" in result, True)

    def test_ler_is_float(self):
        """ler 값은 float이어야 한다"""
        result = run_simulation(**self.base_params)
        self.assertEqual(type(result["ler"]), float)

    def test_ler_in_range(self):
        """ler 값은 0.0 이상 1.0 이하여야 한다"""
        result = run_simulation(**self.base_params)
        self.assertEqual(0.0 <= result["ler"] <= 1.0, True)

    # ── 경계값 ────────────────────────────────────────────────

    def test_zero_noise(self):
        """노이즈가 0이면 ler은 0.0이어야 한다"""
        result = run_simulation(
            distance=3, rounds=3,
            p_gate=0.0, p_meas=0.0, p_leak=0.0,
            shots=100,
        )
        self.assertEqual(result["ler"], 0.0)

    def test_minimum_shots(self):
        """shots=1 최솟값에서도 정상 동작해야 한다"""
        result = run_simulation(
            distance=3, rounds=1,
            p_gate=0.01, p_meas=0.01, p_leak=0.0,
            shots=1,
        )
        self.assertEqual("ler" in result, True)

    def test_high_noise_ler_nonzero(self):
        """높은 노이즈에서는 ler이 0보다 커야 한다"""
        result = run_simulation(
            distance=3, rounds=3,
            p_gate=0.2, p_meas=0.2, p_leak=0.0,
            shots=500,
        )
        self.assertEqual(result["ler"] > 0.0, True)

    def test_p_leak_default(self):
        """p_leak=0.0 명시적으로 전달 시 정상 동작해야 한다"""
        params = dict(self.base_params)
        params["p_leak"] = 0.0
        result = run_simulation(**params)
        self.assertEqual("ler" in result, True)

    def test_larger_distance(self):
        """distance=5에서도 정상 범위의 ler을 반환해야 한다"""
        result = run_simulation(
            distance=5, rounds=3,
            p_gate=0.01, p_meas=0.01, p_leak=0.0,
            shots=50,
        )
        self.assertEqual(0.0 <= result["ler"] <= 1.0, True)


if __name__ == "__main__":
    unittest.main()
