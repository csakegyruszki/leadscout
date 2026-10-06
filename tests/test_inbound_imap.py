"""IMAP poller against a fake imaplib client: no login, no network."""
import imaplib

import pytest
from inbound_support import Runner, fixture_bytes, mail_env  # noqa: F401 - mail_env is a fixture

from leadscout.inbound import imap, service


class FakeIMAP:
    """Just enough of imaplib.IMAP4_SSL. Records every call; UIDs map to raw messages."""
    instances: list = []

    def __init__(self, host, port=993, timeout=None):
        self.host, self.port, self.timeout = host, port, timeout
        self.messages = dict(type(self).mailbox)
        self.log: list = []
        self.copy_ok = True
        type(self).instances.append(self)

    mailbox: dict = {}
    capabilities = ("IMAP4REV1", "UIDPLUS")
    sizes: dict = {}
    fetch_fails = False

    def login(self, user, password):
        self.log.append(("login", user))
        if password == "wrong":
            raise imaplib.IMAP4.error("AUTHENTICATIONFAILED server said: secret-detail")

    def select(self, folder):
        self.log.append(("select", folder))
        return "OK", [b"1"]

    def uid(self, command, *args):
        self.log.append(("uid", command, *args))
        if command == "SEARCH":
            return "OK", [b" ".join(self.messages)]
        if command == "FETCH" and args[1] == "(RFC822.SIZE)":
            uid = args[0].encode() if isinstance(args[0], str) else args[0]
            size = self.sizes.get(uid, len(self.messages[uid]))
            return "OK", [b"1 (UID %s RFC822.SIZE %d)" % (uid, size)]
        if command == "FETCH":
            if self.fetch_fails:
                return "NO", [b""]
            uid = args[0].encode() if isinstance(args[0], str) else args[0]
            return "OK", [(b"1 (UID %s BODY[] {%d}" % (uid, len(self.messages[uid])), self.messages[uid]), b")"]
        if command == "COPY":
            return ("OK" if self.copy_ok else "NO"), [b""]
        return "OK", [b""]

    def expunge(self):
        self.log.append(("expunge",))
        return "OK", [b""]

    def close(self):
        self.log.append(("close",))

    def logout(self):
        self.log.append(("logout",))


@pytest.fixture
def imap_env(monkeypatch, mail_env):
    FakeIMAP.instances = []
    FakeIMAP.mailbox = {b"11": fixture_bytes("plain.eml"), b"12": fixture_bytes("freemail_nosite.eml")}
    monkeypatch.setattr(imaplib, "IMAP4_SSL", FakeIMAP)
    for k in ("HOST", "USER", "PASSWORD", "PORT", "FOLDER", "PROCESSED_FOLDER", "MAX_PER_POLL"):
        monkeypatch.delenv(f"LEADSCOUT_IMAP_{k}", raising=False)
    monkeypatch.setenv("LEADSCOUT_IMAP_HOST", "imap.test")
    monkeypatch.setenv("LEADSCOUT_IMAP_USER", "leads@vendor.test")
    monkeypatch.setenv("LEADSCOUT_IMAP_PASSWORD", "app-password-123")
    runner = Runner()
    monkeypatch.setattr(service, "process_lead", runner)
    return runner


def _calls(client, name):
    return [c for c in client.log if c[0] == "uid" and c[1] == name]


def test_poll_once_fetches_peek_processes_and_marks_seen_after_the_record(imap_env, mail_env):
    result = imap.poll_once()
    assert result == {"counts": {"processed": 1, "needs_review": 1}, "errors": 0, "total": 2}
    c = FakeIMAP.instances[0]
    assert (c.host, c.port) == ("imap.test", 993) and ("select", '"INBOX"') in c.log
    bodies = [call for call in _calls(c, "FETCH") if call[3] != "(RFC822.SIZE)"]
    assert bodies and all(call[3] == "(BODY.PEEK[])" for call in bodies)       # never a plain BODY[]
    assert [call[2] for call in _calls(c, "STORE")] == ["11", "12"]
    assert all(call[3:] == ("+FLAGS", "(\\Seen)") for call in _calls(c, "STORE"))
    assert ("logout",) in c.log
    assert len(list((mail_env.inbound / "records").iterdir())) == 2
    assert len(imap_env.leads) == 1


def test_failed_processing_leaves_the_message_unseen(imap_env, monkeypatch):
    monkeypatch.setattr(service, "process_lead", Runner(exc=RuntimeError("down")))
    result = imap.poll_once()
    assert result["errors"] == 1 and result["counts"]["failed"] == 1
    assert [call[2] for call in _calls(FakeIMAP.instances[0], "STORE")] == ["12"]    # only the needs_review one


def test_processed_folder_copy_then_delete_then_expunge(imap_env, monkeypatch):
    monkeypatch.setenv("LEADSCOUT_IMAP_PROCESSED_FOLDER", "Processed")
    imap.poll_once()
    log = FakeIMAP.instances[0].log
    i_copy = log.index(("uid", "COPY", "11", '"Processed"'))
    assert ("uid", "STORE", "11", "+FLAGS", "(\\Deleted)") in log[i_copy:]
    assert ("uid", "EXPUNGE", "11") in log          # scoped to this message, never a bare EXPUNGE
    assert ("expunge",) not in log


def test_without_uidplus_nothing_is_expunged(imap_env, monkeypatch):
    monkeypatch.setenv("LEADSCOUT_IMAP_PROCESSED_FOLDER", "Processed")
    monkeypatch.setattr(FakeIMAP, "capabilities", ("IMAP4REV1",))
    imap.poll_once()
    log = FakeIMAP.instances[0].log
    assert ("uid", "STORE", "11", "+FLAGS", "(\\Deleted)") in log      # flagged ...
    assert ("expunge",) not in log and not any(c[:2] == ("uid", "EXPUNGE") for c in log)   # ... not removed


def test_oversize_message_is_recorded_without_downloading_its_body(imap_env, monkeypatch, mail_env):
    monkeypatch.setenv("LEADSCOUT_INBOUND_MAX_BYTES", "5000")
    monkeypatch.setattr(FakeIMAP, "sizes", {b"11": 10_000_000})
    result = imap.poll_once()
    c = FakeIMAP.instances[0]
    assert result["counts"].get("rejected") == 1 and result["errors"] == 0
    assert not any(call[2] == "11" and call[3] == "(BODY.PEEK[])" for call in _calls(c, "FETCH"))
    assert "11" in [call[2] for call in _calls(c, "STORE")]                 # acknowledged: a record exists


def test_failed_fetch_counts_as_an_error_and_is_not_acknowledged(imap_env, monkeypatch):
    monkeypatch.setattr(FakeIMAP, "fetch_fails", True)
    result = imap.poll_once()
    assert result["errors"] == 2 and result["counts"] == {"error": 2}
    assert _calls(FakeIMAP.instances[0], "STORE") == []


def test_dead_letter_is_acknowledged_so_the_mailbox_does_not_retry_forever(imap_env, monkeypatch):
    monkeypatch.setenv("LEADSCOUT_INBOUND_MAX_ATTEMPTS", "2")
    monkeypatch.setattr(service, "process_lead", Runner(exc=RuntimeError("down")))
    first = imap.poll_once()
    assert first["counts"]["failed"] == 1
    second = imap.poll_once()
    assert second["counts"]["dead_letter"] == 1
    assert "11" in [call[2] for call in _calls(FakeIMAP.instances[1], "STORE")]
    third = imap.poll_once()                                                # settled: now a plain duplicate
    assert third["counts"].get("failed", 0) == 0


def test_failed_copy_does_not_delete(imap_env, monkeypatch):
    monkeypatch.setenv("LEADSCOUT_IMAP_PROCESSED_FOLDER", "Processed")
    monkeypatch.setattr(FakeIMAP, "uid", _uid_copy_fails(FakeIMAP.uid))
    imap.poll_once()
    log = FakeIMAP.instances[0].log
    assert not any(c[:2] == ("uid", "STORE") and c[-1] == "(\\Deleted)" for c in log) and ("expunge",) not in log


def _uid_copy_fails(original):
    def uid(self, command, *args):
        if command == "COPY":
            self.log.append(("uid", command, *args))
            return "NO", [b""]
        return original(self, command, *args)
    return uid


def test_max_per_poll_limits_the_batch(imap_env, monkeypatch):
    monkeypatch.setenv("LEADSCOUT_IMAP_MAX_PER_POLL", "1")
    assert imap.poll_once()["total"] == 1


def test_second_poll_does_not_rerun_the_pipeline(imap_env):
    imap.poll_once()
    imap.poll_once()                       # the fake mailbox still returns both; the store dedupes them
    assert len(imap_env.leads) == 1


def test_config_requires_host_user_password(monkeypatch, imap_env):
    monkeypatch.delenv("LEADSCOUT_IMAP_PASSWORD")
    with pytest.raises(imap.ImapConfigError):
        imap.poll_once()


def test_login_failure_message_never_contains_the_server_reply_or_password(imap_env, monkeypatch):
    monkeypatch.setenv("LEADSCOUT_IMAP_PASSWORD", "wrong")
    with pytest.raises(imap.ImapConfigError) as exc:
        imap.poll_once()
    text = str(exc.value) + repr(exc.value.__cause__)
    assert "wrong" not in text and "secret-detail" not in text


def test_config_repr_hides_the_password():
    cfg = imap.ImapConfig(host="h", user="u", password="topsecret")
    assert "topsecret" not in repr(cfg) and "topsecret" not in str(cfg)


def test_cli_mail_poll_exit_codes(imap_env, monkeypatch, capsys):
    from leadscout import cli
    assert cli.main(["mail", "poll", "--once"]) == 0
    monkeypatch.delenv("LEADSCOUT_IMAP_HOST")
    assert cli.main(["mail", "poll", "--once"]) == 2
    assert "LEADSCOUT_IMAP_HOST" in capsys.readouterr().err
