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
    parser = argparse.ArgumentParser(description="MLflow registry alias 승격")
    parser.add_argument("--name", default="mlp_d3", help="registered model 이름")
    parser.add_argument("--from-alias", default="challenger")
    parser.add_argument("--to-alias", default="champion")
    args = parser.parse_args()

    uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlruns.db")
    mlflow.set_tracking_uri(uri)
    client = MlflowClient()

    cand = client.get_model_version_by_alias(args.name, args.from_alias)
    client.set_registered_model_alias(args.name, args.to_alias, cand.version)
    print(f"promoted {args.name} v{cand.version}: @{args.from_alias} -> @{args.to_alias}")


if __name__ == "__main__":
    main()
