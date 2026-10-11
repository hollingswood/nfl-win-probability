import pytest

ALERT_ENV = ("NTFY_TOPIC", "PUSHOVER_USER", "PUSHOVER_TOKEN", "TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID")


@pytest.fixture(autouse=True)
def _no_real_alerts(monkeypatch):
    """Tests never reach a real phone: the workflows export the alert secrets, so clear them for every test
    (a test that needs a channel sets it and stubs notify.send / _post itself)."""
    for k in ALERT_ENV:
        monkeypatch.delenv(k, raising=False)
