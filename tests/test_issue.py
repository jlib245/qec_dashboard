# tests/test_issue.py
import unittest
from unittest.mock import patch, MagicMock

from app.issue import create_github_issue


class TestCreateGithubIssue(unittest.TestCase):
    """create_github_issue() 테스트. requests/os.environ를 mock으로 격리한다."""

    def setUp(self):
        self.logger = MagicMock()

    @patch("app.issue.requests.post")
    @patch.dict("os.environ", {}, clear=True)
    def test_skips_when_env_missing(self, mock_post):
        """GH_REPO/GH_TOKEN 없으면 POST 없이 warning만 남기고 스킵한다"""
        create_github_issue("t", "b", self.logger)
        mock_post.assert_not_called()
        self.logger.warning.assert_called()

    @patch("app.issue.requests.post")
    @patch.dict("os.environ", {"GH_REPO": "me/repo", "GH_TOKEN": "tok"}, clear=True)
    def test_posts_when_env_set(self, mock_post):
        """env가 있으면 올바른 URL/헤더/payload로 POST 한다"""
        mock_post.return_value = MagicMock(status_code=201)
        create_github_issue("My Title", "My Body", self.logger)

        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        self.assertEqual(args[0], "https://api.github.com/repos/me/repo/issues")
        self.assertEqual(kwargs["json"], {"title": "My Title", "body": "My Body"})
        self.assertEqual(kwargs["headers"]["Authorization"], "Bearer tok")

    @patch("app.issue.requests.post")
    @patch.dict("os.environ", {"GH_REPO": "me/repo", "GH_TOKEN": "tok"}, clear=True)
    def test_warns_on_api_error(self, mock_post):
        """GitHub API가 에러 상태코드를 주면 warning을 남긴다"""
        mock_post.return_value = MagicMock(status_code=403, text="forbidden")
        create_github_issue("t", "b", self.logger)
        self.logger.warning.assert_called()


if __name__ == "__main__":
    unittest.main()
