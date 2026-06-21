# app/issue.py
import os

import requests


def create_github_issue(title: str, body: str, logger) -> None:
    """에러 발생 시 GitHub Issue를 자동 생성한다 (수업 SpamCheck 패턴).

    GH_REPO("owner/repo")와 GH_TOKEN 환경변수가 모두 있어야 동작하고,
    없으면 조용히 스킵한다(로컬/CI/테스트에서 안전).
    """
    repo = os.getenv("GH_REPO")
    token = os.getenv("GH_TOKEN")
    if not repo or not token:
        # 토큰 없으면 조용히 스킵(수업/실습에서 편함)
        logger.warning("GH_REPO/GH_TOKEN not set; skipping GitHub issue creation.")
        return

    url = f"https://api.github.com/repos/{repo}/issues"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
    }
    payload = {"title": title, "body": body}
    r = requests.post(url, headers=headers, json=payload, timeout=10)
    if r.status_code >= 300:
        logger.warning(f"Failed to create issue: {r.status_code} {r.text[:200]}")
