# ml/promote.py
import os
import argparse

import mlflow
from mlflow import MlflowClient


def main():
    """challenger 모델을 champion으로 승격한다 (서빙 model_loader는 champion만 로드).

    검증을 통과한 모델에만 champion alias를 옮기는 단순 거버넌스 단계.
    필요하면 challenger metric vs 현재 champion 비교 로직을 여기에 추가한다.
    """
    parser = argparse.ArgumentParser(description="MLflow registry alias 승격 / 롤백")
    parser.add_argument("--name", default="mlp_d3_r3", help="registered model 이름")
    parser.add_argument("--from-alias", default="challenger")
    parser.add_argument("--to-alias", default="champion")
    parser.add_argument(
        "--version", default=None,
        help="지정 시 to-alias를 이 버전으로 직접 이동 (롤백용). 미지정 시 from-alias의 버전으로 승격.",
    )
    args = parser.parse_args()

    uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlruns.db")
    mlflow.set_tracking_uri(uri)
    client = MlflowClient()

    if args.version is not None:
        # 롤백/지정: to-alias를 특정 버전으로 이동
        version = args.version
        client.set_registered_model_alias(args.name, args.to_alias, version)
        print(f"set {args.name}@{args.to_alias} -> v{version} (rollback/pin)")
    else:
        # 승격: from-alias가 가리키는 버전을 to-alias로
        cand = client.get_model_version_by_alias(args.name, args.from_alias)
        client.set_registered_model_alias(args.name, args.to_alias, cand.version)
        print(f"promoted {args.name} v{cand.version}: @{args.from_alias} -> @{args.to_alias}")


if __name__ == "__main__":
    main()
