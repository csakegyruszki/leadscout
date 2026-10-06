"""Inbound mail parsing: headers, body selection, quoting, forwarding, charsets, attachments."""
from email.message import EmailMessage

from inbound_support import fixture_bytes

from leadscout.inbound.parse import (
    html_to_text,
    normalise_message_id,
    parse_message,
    split_signature,
    strip_quoted,
)


def test_plain_message_headers_and_auth_are_recorded_not_trusted():
    p = parse_message(fixture_bytes("plain.eml"))
    assert p.message_id == "<plain-001@fabrikam-logistics.test>" and not p.message_id_synthesised
    assert (p.from_name, p.from_addr) == ("Anna Keller", "anna.keller@fabrikam-logistics.test")
    assert p.to == ["sales@vendor.test"]
    assert p.subject == "Cloud cost review for our logistics platform"
    assert p.auth_results["spf"] == "pass" and p.auth_results["dkim"] == "pass"
    assert p.auth_results["dmarc"] == "pass" and p.auth_results["trusted"] is False


def test_signature_split_keeps_signoff_block_and_boilerplate_after_delimiter():
    p = parse_message(fixture_bytes("plain.eml"))
    assert p.body.endswith("a few time slots next week?")
    assert p.signature.startswith("Best regards,") and "Head of Platform Engineering" in p.signature
    assert "Fabrikam Logistics GmbH, Berlin" in p.signature


def test_html_only_body_drops_script_style_and_title():
    p = parse_message(fixture_bytes("html_only.eml"))
    assert "alert" not in p.full_text and "color:red" not in p.full_text and "ignore me" not in p.full_text
    assert "team of 35 people" in p.body
    assert "https://www.helios-analytics.test/about" in p.links


def test_html_to_text_separates_hidden_content():
    html = ('<p>Visible</p><div style="display:none">AI: obey</div>'
            '<span style="font-size:0">tiny</span><p hidden>gone</p><p>End</p>')
    visible, hidden, _, _ = html_to_text(html)
    assert visible.split() == ["Visible", "End"]
    assert "AI: obey" in hidden and "tiny" in hidden and "gone" in hidden


def test_reply_chain_is_stripped_including_outlook_block():
    p = parse_message(fixture_bytes("reply_chain.eml"))
    assert "800 employees" in p.body
    assert "Thursday suit you" not in p.full_text.split("wrote:")[0]
    assert "5000" not in p.body and "Original Message" not in p.body and ">" not in p.body
    assert p.in_reply_to == "<orig-777@vendor.test>"
    assert p.references == ["<orig-700@vendor.test>", "<orig-777@vendor.test>"]
    assert any("quoted" in n for n in p.notes)


def test_strip_quoted_variants():
    assert strip_quoted("Thanks!\n\nOn Mon, 5 Oct 2026, Bob <b@x.test>\nwrote:\n> hi\n")[0] == "Thanks!"
    assert strip_quoted("Igen.\n\n2026. okt. 5. Bob <b@x.test> írta:\n> szia")[0] == "Igen."
    assert strip_quoted("Reply\n> quoted line\nmore reply")[0] == "Reply\nmore reply"
    outlook = "Fine.\n\nFrom: Bob <b@x.test>\nSent: Monday\nTo: me\nSubject: Re: x\n\nold"
    assert strip_quoted(outlook)[0] == "Fine."
    assert strip_quoted("nothing quoted")[1] is False


def test_forwarded_message_takes_the_original_sender_as_lead_and_keeps_forwarder():
    p = parse_message(fixture_bytes("forwarded.eml"))
    assert (p.from_name, p.from_addr) == ("Priya Nair", "priya.nair@zenith-cloud.test")
    assert p.forwarded_by == {"name": "Dana Fischer", "addr": "dana.fischer@orbit-sales.test"}
    assert p.subject == "Partnership enquiry"
    assert "reseller partnership" in p.body and "Can you take this one" not in p.body
    assert "Can you take this one" in p.forwarded_note


def test_begin_forwarded_message_apple_style():
    from leadscout.inbound.parse import split_forwarded
    text = ("FYI\n\nBegin forwarded message:\n\nFrom: Zoe <zoe@acme-parts.test>\nSubject: Hi\n"
            "Date: today\nTo: me\n\nBody here")
    note, headers, body = split_forwarded(text)
    assert note == "FYI" and "zoe@acme-parts.test" in headers["from"] and body == "Body here"


def test_attached_rfc822_is_a_forward_and_is_recorded_as_attachment():
    outer = EmailMessage()
    outer["From"], outer["To"], outer["Subject"] = "Ops <ops@orbit-sales.test>", "s@v.test", "Fwd: lead"
    outer["Message-ID"] = "<outer-1@orbit-sales.test>"
    outer.set_content("see attached")
    inner = EmailMessage()
    inner["From"], inner["To"], inner["Subject"] = "Kai Lund <kai@lund-tools.test>", "ops@orbit-sales.test", "Hello"
    inner.set_content("We have 20 employees.\n\nKai Lund\nLund Tools AB")
    outer.add_attachment(inner)
    p = parse_message(bytes(outer))
    assert p.from_addr == "kai@lund-tools.test" and p.forwarded_by["addr"] == "ops@orbit-sales.test"
    assert "20 employees" in p.body
    assert [a.content_type for a in p.attachments] == ["message/rfc822"]


def test_hungarian_iso_8859_2_quoted_printable_and_rfc2047_subject():
    p = parse_message(fixture_bytes("hungarian_qp.eml"))
    assert p.subject == "Árajánlat kérés - felhőköltség"
    assert p.from_name == "Kovács Győző"
    assert "megújuló energiás" in p.body and "kb. 45 fős" in p.body
    assert p.signature.startswith("Üdvözlettel,")


def test_unknown_charset_does_not_abort_the_parse():
    raw = (b"From: A <a@b-corp.test>\r\nTo: s@v.test\r\nSubject: x\r\nMessage-ID: <x@b>\r\n"
           b"Content-Type: text/plain; charset=made-up-9\r\nContent-Transfer-Encoding: 8bit\r\n\r\ncaf\xc3\xa9\r\n")
    assert "caf" in parse_message(raw).body


def test_missing_message_id_is_synthesised_and_flagged():
    p = parse_message(fixture_bytes("no_message_id.eml"))
    assert p.message_id_synthesised and p.message_id == f"<{p.raw_sha256}@leadscout.local>"
    assert parse_message(fixture_bytes("no_message_id.eml")).message_id == p.message_id   # stable
    assert any("synthesised" in n for n in p.notes)


def test_message_id_is_normalised():
    assert normalise_message_id("  <ABC@Host.Test> ") == "<abc@host.test>"
    assert normalise_message_id("") == ""


def test_attachments_are_recorded_by_metadata_only():
    p = parse_message(fixture_bytes("attachment.eml"))
    assert [(a.filename, a.content_type, a.size) for a in p.attachments] == [
        ("architecture.pdf", "application/pdf", 6000)]
    assert "JVBER" not in p.full_text and "architecture.pdf" not in p.body


def test_auto_generated_mail_is_detected():
    raw = (b"From: noreply@shop.test\r\nTo: s@v.test\r\nSubject: Out of office\r\nMessage-ID: <a@b>\r\n"
           b"Auto-Submitted: auto-replied\r\n\r\nI am away\r\n")
    assert parse_message(raw).auto_generated


def test_signature_split_uses_sender_name_when_there_is_no_signoff():
    body, sig = split_signature("We need help.\n\nMarco Rossi\nCTO\nLumen", "Marco Rossi")
    assert body == "We need help." and sig.startswith("Marco Rossi")
