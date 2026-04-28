# Dockerfile
FROM python:3.10-slim

WORKDIR /app

# git: qec-sim을 git+https URL로 받기 위해 필요
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

# 1) CPU-only torch (풀 휠 ~2GB → CPU 휠 ~150MB)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# 2) qec-sim (--no-deps: intel-extension/jupyter/pandas 등 import 체인에 없는 무거운 deps 회피)
RUN pip install --no-cache-dir --no-deps git+https://github.com/jlib245/qec.git

# 3) qec-sim 런타임 + 대시보드 deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 앱 코드
COPY app/ ./app/
COPY static/ ./static/

# Render는 PORT 환경변수로 동적 포트 주입. 로컬 기본값 10000.
ENV PORT=10000
EXPOSE 10000

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
