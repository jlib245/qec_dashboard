# app/logging_config.py
import logging

#   시간 | 레벨 | 파일:라인(함수) | 메시지
_LOG_FORMAT = (
    "%(asctime)s | %(levelname)s | "
    "%(filename)s:%(lineno)d (%(funcName)s) | "
    "%(message)s"
)


def setup_logging(level: int = logging.INFO) -> None:
    """앱 진입점에서 1회 호출. 루트 로거에 포맷/레벨을 설정한다."""
    logging.basicConfig(level=level, format=_LOG_FORMAT)


def get_logger(name: str = "qec_dashboard") -> logging.Logger:
    """이름 있는 로거 반환 (엔드포인트별 호출 추적용)."""
    return logging.getLogger(name)
