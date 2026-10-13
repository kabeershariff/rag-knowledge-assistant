import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "live: needs GEMINI_API_KEY and network access")
