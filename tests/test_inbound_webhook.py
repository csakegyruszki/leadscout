"""POST /inbound/mail: three payload shapes, token auth, duplicates, size cap. Offline."""
import base64
import json

import pytest
from fastapi.testclient import TestClient
from inbound_support import Runner, fixture_bytes, mail_env  # noqa: F401 - mail_env is a fixture

from leadscout import api
from leadscout.inbound import service

TOKEN = "t0ken-for-tests-only"


@pytest.fixture
def client(mail_env, monkeypatch):
    monkeypatch.setenv("LEADSCOUT_INBOUND_TOKEN", TOKEN)
    runner = Runner()
    monkeypatch.setattr(service, "process_lead", runner)
    c = TestClient(api.app)
    c.runner, c.env = runner, mail_env
    return c


def _hdr(token=TOKEN):
    return {"x-leadscout-token": token}


def _records(client):
    d = client.env.inbound / "records"
    return sorted(d.iterdir()) if d.exists() else []


def test_disabled_with_404_when_no_token_is_configured(mail_env, monkeypatch):
    monkeypatch.delenv("LEADSCOUT_INBOUND_TOKEN", raising=False)
    c = TestClient(api.app)
    r = c.post("/inbound/mail", content=fixture_bytes("plain.eml"),
               headers={"content-type": "message/rfc822", "x-leadscout-token": "anything"})
    assert r.status_code == 404


@pytest.mark.parametrize("headers", [{}, {"x-leadscout-token": "wrong"}, {"x-leadscout-token": TOKEN[:-1]}])
def test_missing_or_wrong_token_is_401(client, headers):
    r = client.post("/inbound/mail", content=fixture_bytes("plain.eml"),
                    headers={"content-type": "message/rfc822", **headers})
    assert r.status_code == 401 and not client.runner.leads and _records(client) == []


def test_raw_rfc822_post_returns_202_and_runs_the_pipeline_in_the_background(client):
    r = client.post("/inbound/mail", content=fixture_bytes("plain.eml"),
                    headers={"content-type": "message/rfc822", **_hdr()})
    assert r.status_code == 202
    body = r.json()
    assert body["status"] == "accepted" and body["duplicate"] is False
    assert body["message_id"] == "<plain-001@fabrikam-logistics.test>"
    assert len(client.runner.leads) == 1 and client.runner.leads[0].company == "Fabrikam Logistics GmbH"
    rec = json.loads((client.env.inbound / "records" / f"{body['id']}.json").read_text(encoding="utf-8"))
    assert rec["source"] == "webhook" and rec["status"] == "processed"


def test_token_in_query_string_is_accepted_for_providers_that_cannot_set_headers(client):
    r = client.post(f"/inbound/mail?token={TOKEN}", content=fixture_bytes("plain.eml"),
                    headers={"content-type": "message/rfc822"})
    assert r.status_code == 202
    assert client.post("/inbound/mail?token=nope", content=b"x",
                       headers={"content-type": "message/rfc822"}).status_code == 401


def test_mailgun_multipart_body_mime_field(client):
    r = client.post("/inbound/mail", headers=_hdr(),
                    files={"body-mime": (None, fixture_bytes("forwarded.eml")), "sender": (None, b"x@y.test")})
    assert r.status_code == 202 and r.json()["message_id"] == "<fwd-001@orbit-sales.test>"
    assert client.runner.leads[0].email == "priya.nair@zenith-cloud.test"


def test_sendgrid_raw_email_field_multipart_and_urlencoded(client):
    r1 = client.post("/inbound/mail", headers=_hdr(), files={"email": (None, fixture_bytes("html_only.eml"))})
    assert r1.status_code == 202 and r1.json()["message_id"] == "<html-001@helios-analytics.test>"
    r2 = client.post("/inbound/mail", headers=_hdr(),
                     data={"email": fixture_bytes("hungarian_qp.eml").decode("latin-1")})
    assert r2.status_code == 202 and r2.json()["message_id"] == "<hu-001@napszel-energia.test>"
    assert len(client.runner.leads) == 2


def test_multipart_without_a_raw_field_is_400(client):
    r = client.post("/inbound/mail", headers=_hdr(), files={"subject": (None, b"hello")})
    assert r.status_code == 400 and r.json()["error"] == "bad_payload"


def test_postmark_json_payload(client):
    payload = {
        "FromFull": {"Email": "Priya.Nair@zenith-cloud.test", "Name": "Priya Nair"},
        "From": "priya.nair@zenith-cloud.test", "To": "sales@vendor.test", "Subject": "Partnership enquiry",
        "MessageID": "11111111-2222-3333-4444-555555555555",
        "TextBody": ("Zenith Cloud has around 300 staff.\n\nKind regards,\nPriya Nair\n"
                     "VP Partnerships\nZenith Cloud Ltd"),
        "HtmlBody": "<p>ignored when text exists</p>",
        "Headers": [{"Name": "Message-ID", "Value": "<pm-001@zenith-cloud.test>"},
                    {"Name": "Authentication-Results", "Value": "mx; spf=pass; dkim=fail"}],
        "Attachments": [{"Name": "deck.pdf", "ContentType": "application/pdf",
                         "Content": base64.b64encode(b"%PDF-1.4 fake").decode()}],
    }
    r = client.post("/inbound/mail", json=payload, headers=_hdr())
    assert r.status_code == 202 and r.json()["message_id"] == "<pm-001@zenith-cloud.test>"
    lead = client.runner.leads[0]
    assert (lead.email, lead.company, lead.job_title) == ("priya.nair@zenith-cloud.test", "Zenith Cloud Ltd",
                                                          "VP Partnerships")
    rec = json.loads(_records(client)[0].read_text(encoding="utf-8"))
    assert rec["auth_results"]["dkim"] == "fail" and rec["attachments"][0]["filename"] == "deck.pdf"


def test_postmark_html_only_and_missing_header_message_id(client):
    payload = {"From": "tom@helios-analytics.test", "Subject": "Hi", "MessageID": "abc-123",
               "HtmlBody": "<p>We have 35 people.</p><p>Tom<br>CTO<br>Helios Analytics s.r.o.</p>"}
    r = client.post("/inbound/mail", json=payload, headers=_hdr())
    assert r.status_code == 202 and r.json()["message_id"] == "<abc-123@inbound.postmarkapp.com>"


def test_non_postmark_json_is_400_and_unknown_type_is_400(client):
    assert client.post("/inbound/mail", json={"hello": "world"}, headers=_hdr()).status_code == 400
    r = client.post("/inbound/mail", content=b"x", headers={"content-type": "application/pdf", **_hdr()})
    assert r.status_code == 400


def test_duplicate_delivery_returns_202_with_the_earlier_id_and_no_second_run(client):
    h = {"content-type": "message/rfc822", **_hdr()}
    first = client.post("/inbound/mail", content=fixture_bytes("plain.eml"), headers=h).json()
    second = client.post("/inbound/mail", content=fixture_bytes("plain.eml"), headers=h)
    assert second.status_code == 202
    assert second.json()["duplicate"] is True and second.json()["id"] == first["id"]
    assert len(client.runner.leads) == 1


def test_oversize_request_body_is_413_before_anything_is_stored(client, monkeypatch):
    monkeypatch.setenv("LEADSCOUT_INBOUND_MAX_BYTES", "1000")
    r = client.post("/inbound/mail", content=b"x" * (1000 * 2 + 70 * 1024),
                    headers={"content-type": "message/rfc822", **_hdr()})
    assert r.status_code == 413 and _records(client) == []


def test_message_over_the_cap_but_under_the_body_limit_is_recorded_as_rejected(client, monkeypatch):
    monkeypatch.setenv("LEADSCOUT_INBOUND_MAX_BYTES", "2000")
    r = client.post("/inbound/mail", content=fixture_bytes("attachment.eml"),
                    headers={"content-type": "message/rfc822", **_hdr()})
    assert r.status_code == 202 and r.json()["status"] == "rejected" and not client.runner.leads


def test_pipeline_error_in_background_task_is_recorded_as_failed_not_a_500(client, monkeypatch):
    monkeypatch.setattr(service, "process_lead", Runner(exc=RuntimeError("boom")))
    r = client.post("/inbound/mail", content=fixture_bytes("plain.eml"),
                    headers={"content-type": "message/rfc822", **_hdr()})
    assert r.status_code == 202
    assert json.loads(_records(client)[0].read_text(encoding="utf-8"))["status"] == "failed"
