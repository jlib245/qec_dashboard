# ml/promote.py
import os
import argparse

import mlflow
from mlflow import MlflowClient


def _metric(client, name, version, metric):
    """모델 version의 run에서 metric 값 (없으면 None)."""
    try:
        mv = client.get_model_version(name, str(version))
        return client.get_run(mv.run_id).data.metrics.get(metric)
    except Exception:
        return None


def main():
    """MLflow registry alias 승격(metric 비교) / 롤백.

    승격(challenger→champion): challenger의 test_ler이 현재 champion보다 낮을 때만.
        → "held-out 성능이 가장 좋은 모델이 champion(서빙)"을 코드로 보장.
    롤백(--version): test 비교 없이 to-alias를 특정 버전으로 강제 이동.
    """
    parser = argparse.ArgumentParser(description="MLflow registry alias 승격/롤백")
    parser.add_argument("--name", default="mlp_d3_r3", help="registered model 이름")
    parser.add_argument("--from-alias", default="challenger")
    parser.add_argument("--to-alias", default="champion")
    parser.add_argument(
        "--version", default=None,
        help="지정 시 to-alias를 이 버전으로 직접 이동 (롤백/핀, metric 비교 생략).",
    )
    parser.add_argument("--metric", default="test_ler", help="승격 판정 metric (낮을수록 좋음)")
    parser.add_argument("--force", action="store_true", help="metric 비교 무시하고 강제 승격")
    args = parser.parse_args()

    uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlruns.db")
    mlflow.set_tracking_uri(uri)
    client = MlflowClient()

    # 롤백/핀: 검증 없이 alias 이동
    if args.version is not None:
        client.set_registered_model_alias(args.name, args.to_alias, args.version)
        print(f"set {args.name}@{args.to_alias} -> v{args.version} (rollback/pin)")
        return

    cand = client.get_model_version_by_alias(args.name, args.from_alias)
    cand_m = _metric(client, args.name, cand.version, args.metric)

    # 현재 champion (없으면 무조건 승격)
    cur = cur_m = None
    try:
        cur = client.get_model_version_by_alias(args.name, args.to_alias)
        cur_m = _metric(client, args.name, cur.version, args.metric)
    except Exception:
        pass

    # metric 게이트 (낮을수록 좋음). 둘 다 값이 있고 --force가 아닐 때만 막는다.
    if cur is not None and cand_m is not None and cur_m is not None and not args.force:
        if cand_m >= cur_m:
            print(
                f"승격 거부: {args.from_alias}(v{cand.version}) {args.metric}={cand_m:.4f} "
                f">= {args.to_alias}(v{cur.version}) {cur_m:.4f} — 더 좋지 않음. 강제: --force"
            )
            return

    client.set_registered_model_alias(args.name, args.to_alias, cand.version)
    cmp = (
        f" ({args.metric}: {cand_m:.4f} < {cur_m:.4f})"
        if cand_m is not None and cur_m is not None else ""
    )
    print(f"promoted {args.name} v{cand.version}: @{args.from_alias} -> @{args.to_alias}{cmp}")


if __name__ == "__main__":
    main()
