# Inbound e-mail

LeadScout can start from a raw e-mail instead of a form post: a message arrives by IMAP polling or
by webhook, is parsed and de-duplicated, becomes a `Lead`, goes through `process_lead`, and leaves
a JSON record plus (optionally) a reply **draft**. Nothing is sent unless you switch sending on.

## Architecture

```
IMAP poller (imap.py) ──┐
webhook POST /inbound/mail ─┤──> service.prepare:  size cap -> parse.py -> store.py claim (dedupe)
CLI `mail process FILE` ──┘                              │ duplicate / rejected: answered here
                                                         ▼
                          service.run_prepared:  extract.py (deterministic, optional LLM fill-in, injection flags)
                                                 -> pipeline.process_lead (skipped when the lead is unusable or flagged)
                                                 -> out/inbound/records/<id>.json
                                                 -> out/inbound/drafts/<id>.eml  (only when allowed)
                                                 -> SMTP send (needs the full reply gate, see Security)
```

* The webhook answers `202` after parse + dedupe (done in a worker thread, off the event loop); the pipeline runs
  in a FastAPI background task. Extraction + pipeline runs are bounded by a semaphore
  (`LEADSCOUT_INBOUND_CONCURRENCY`, default 1: the pipeline drains a process-global LLM telemetry list, so two
  concurrent runs in one process would mix their call counts and costs).
* The raw message is written to `out/inbound/raw/<id>.eml` when it is claimed, before the webhook answers. If the
  process dies afterwards, `leadscout mail resume` re-drives it from that copy.
* `store.py` is one SQLite file (`out/inbound/inbound.sqlite3`). A message is claimed (primary key
  Message-ID; secondary key a hash of normalised sender + subject + body) inside a write transaction
  *before* any work starts, so concurrent retries have exactly one winner. A claim left `claimed`
  for more than 15 minutes, or a record that ended `failed`, can be claimed again - at most
  `LEADSCOUT_INBOUND_MAX_ATTEMPTS` times (default 3); the last failed attempt is recorded as the terminal
  `dead_letter`, which the IMAP poller acknowledges (marks `\Seen`) so the mailbox stops retrying it.
* The pipeline runner is a parameter of `service.process_message`, so tests never touch research,
  the LLM or the network.

## Record JSON (`out/inbound/records/<id>.json`)

`<id>` is the first 24 hex chars of sha256(Message-ID). Top-level keys:

| key | content |
|---|---|
| `record_id`, `status`, `reason`, `source`, `processed_at` | `status`: `processed` · `needs_review` (no usable website/company, or injection flags: pipeline not run) ·
`rejected` · `failed` (will be retried) · `dead_letter` (gave up) · `extracted` (`--no-pipeline`; written to
`<id>.dryrun.json`, never over a real record). `source`: `imap`, `webhook`, `file:<name>`. |
| `mail` | `message_id`, `message_id_synthesised`, `from_name`, `from_addr`, `reply_to`, `to`, `subject`, `date`, `in_reply_to`, `references`, `forwarded_by` (`{name, addr}` or null), `size_bytes`, `raw_sha256`, `auto_generated` |
| `body_preview` | first 500 chars of the sender's own text (quoted history and signature removed) |
| `content_hash` | the secondary dedupe key |
| `lead` | `name`, `email`, `company`, `website`, `job_title`, `company_size_band` as handed to the pipeline |
| `extraction` | `methods` (per field: `from_header`, `forwarded_original_sender`, `sender_domain`, `signature_or_body_url`, `signature_legal_form`, `website_domain_label`, `labelled_in_body`, `signature_title_word`, `stated_headcount_regex`, `llm_fill_in`), `llm_used`, `llm_note`, `missing`, `notes` |
| `injection_flags` | names of the heuristics that fired (see Security) |
| `auth_results` | `spf` / `dkim` / `dmarc` verdicts as written in `Authentication-Results`, the raw header, and `"trusted": false` |
| `attachments` | `[{filename, content_type, size}]` - metadata only |
| `outcome` | `LeadOutcome.to_dict()`, or null when the pipeline did not run |
| `next_action` | `notify.next_action(outcome)` |
| `draft` | `{path, suppressed_reason, sent}`; `send_blocked_reason` says which condition of the reply gate failed; `send_error` when a send was attempted and failed |

A duplicate is not a new record: the call returns `{record_id: <earlier id>, status: "duplicate",
duplicate_of, reason}`.

## Environment variables

| variable | default | meaning |
|---|---|---|
| `LEADSCOUT_IMAP_HOST` / `_USER` / `_PASSWORD` | - | mailbox to poll; all three required for `leadscout mail poll` |
| `LEADSCOUT_IMAP_PORT` | `993` | IMAP over TLS (`IMAP4_SSL`) |
| `LEADSCOUT_IMAP_FOLDER` | `INBOX` | folder searched for `UNSEEN` |
| `LEADSCOUT_IMAP_PROCESSED_FOLDER` | empty | if set: after a record is written, the message is copied there, flagged `Deleted` and removed with `UID EXPUNGE` (servers advertising UIDPLUS only; otherwise it stays flagged, never a bare `EXPUNGE`) |
| `LEADSCOUT_IMAP_MAX_PER_POLL` | `25` | messages handled per pass |
| `LEADSCOUT_INBOUND_TOKEN` | unset | shared secret for the webhook; **unset = the endpoint answers 404** |
| `LEADSCOUT_INBOUND_MAX_BYTES` | `10485760` | raw message size cap; larger messages are rejected with a recorded reason |
| `LEADSCOUT_INBOUND_CONCURRENCY` | `1` | concurrent extraction + pipeline runs |
| `LEADSCOUT_INBOUND_MAX_ATTEMPTS` | `3` | failed attempts before `dead_letter` |
| `LEADSCOUT_SEND_REPLIES` | off | must be on **in addition to** `LEADSCOUT_SEND_EMAIL` and `SMTP_*` (and `LEADSCOUT_PROOF_MODE` off) before any reply is sent; `LEADSCOUT_SEND_EMAIL` alone only enables rep notifications |
| `LEADSCOUT_ORG_PROFILE` | `default` | profile whose `identity` / `reply` sections shape the draft |

## IMAP recipe

1. Use a dedicated mailbox. Create an **app password** (Gmail / Microsoft 365 / Fastmail all offer one; the account
   password should not be used) and put it in `.env`:
   `LEADSCOUT_IMAP_HOST=imap.example.org`, `LEADSCOUT_IMAP_USER=leads@example.org`, `LEADSCOUT_IMAP_PASSWORD=...`.
2. Try one pass by hand: `leadscout mail poll --once` (or `python -m leadscout.cli mail poll --once`).
   Exit codes: `0` ok, `1` at least one message failed (left unseen, retried next pass), `2` configuration/login error.
3. Install the timer (paths in the units assume `/opt/leadscout`, a venv at `.venv`, and a `leadscout` user -
   edit them):
   ```
   sudo cp infra/systemd/leadscout-mail-poll.* /etc/systemd/system/
   sudo systemctl daemon-reload && sudo systemctl enable --now leadscout-mail-poll.timer
   journalctl -u leadscout-mail-poll.service -n 50
   ```
   The units name the `.env` file via `EnvironmentFile=`; no secret appears in them. The service runs
   `leadscout mail poll --once --resume`: one IMAP pass, then the resume pass. For a webhook-only deployment
   change `ExecStart` to `leadscout mail resume`.

`leadscout mail resume` re-drives messages that were received but never got a result: `claimed` rows older than
15 minutes (the process died after the claim, e.g. after the webhook's 202) and `failed` rows with attempts left.
It works from `out/inbound/raw/<id>.eml`; a row with no raw copy becomes `dead_letter`. Exit code 1 if any message
ended `failed` or `dead_letter`.

The advertised `RFC822.SIZE` is checked first: a message over the cap is recorded as `rejected` without downloading its body. A failed fetch counts as an error and is not acknowledged.

Messages are fetched with `BODY.PEEK[]` and marked `\Seen` only after the record file exists, so a crash
mid-message leaves it unread for the next pass (the store then recognises it if it had already been claimed).
The password is read once for `login()` and is not logged; a failed login raises a message that omits the
server's reply.

## Webhook recipes

All of them POST to `https://<host>/inbound/mail` with the token in the header `X-LeadScout-Token` or - for
providers that cannot set headers - as `?token=<value>` (it will then appear in access logs). Responses:
`202` `{id, status: accepted|duplicate|rejected, duplicate, message_id}`; `400` unreadable payload; `401` bad
token; `404` webhook disabled; `413` request body over ~1.5x the message cap.

**Cloudflare Email Routing + Email Worker** (raw MIME):

```js
export default {
  async email(message, env) {
    const raw = await new Response(message.raw).arrayBuffer();
    const res = await fetch(env.LEADSCOUT_URL + "/inbound/mail", {
      method: "POST",
      headers: { "content-type": "message/rfc822", "x-leadscout-token": env.LEADSCOUT_TOKEN },
      body: raw,
    });
    if (!res.ok) throw new Error("leadscout webhook " + res.status);   // surfaces in Worker logs
  },
};
```

Set `LEADSCOUT_URL` as a variable and `LEADSCOUT_TOKEN` as a secret on the Worker, and point a custom address in
Email Routing at it. (Not run against Cloudflare in this repo; what the platform does after a thrown error is
not verified here.)

**Mailgun**: route action `forward("https://<host>/inbound/mail/mime?token=<value>")`. The route ending in
`/mime` is the form in which Mailgun posts the full message as field `body-mime`; `/inbound/mail/mime` is an alias
of `/inbound/mail`. (Based on Mailgun's documentation as recalled; not verified against a live route.)

**SendGrid Inbound Parse**: tick "POST the raw, full MIME message"; destination URL
`https://<host>/inbound/mail?token=<value>`. The message arrives in the form field `email`.

**Postmark inbound**: set the inbound webhook URL to `https://<host>/inbound/mail?token=<value>`. The JSON
(`FromFull`, `To`, `Subject`, `TextBody`, `HtmlBody`, `Headers`, `Attachments`) is rebuilt into a message; the
sender's own `Message-ID` is taken from `Headers`, falling back to `<MessageID@inbound.postmarkapp.com>`.

Multipart and url-encoded forms are parsed with the standard library; `python-multipart` is not a dependency.

## What the parser does

* Body: `text/plain` first, else HTML converted to text (script/style/head dropped; hidden elements collected
  separately for the injection check). Quoted history (`>` lines, "On ... wrote:", `-----Original Message-----`,
  Outlook `From:/Sent:` blocks and equivalents in HU/DE/FR) and the signature (`-- ` or a sign-off line) are split off.
* Forwarded mail (`---------- Forwarded message ---------`, `Begin forwarded message:`, an attached `message/rfc822`):
  the **original** sender is the lead, the forwarder is kept in `mail.forwarded_by`, and a reply draft goes to the
  original sender.
* Lead fields, deterministic first: e-mail and name from the sender; website from the sender's domain unless it
  is a free-mail domain, else from the signature, then body links (shorteners, tracking, social and meeting links
  skipped); company from a legal-form line in the signature, a `Company:` label, or the domain; job title and headcount
  band by pattern. An LLM may then fill fields that are still empty - see Security.
* No usable website/company -> `needs_review`, pipeline not run.

## Security notes

* **The From address is spoofable.** `Authentication-Results` is copied into the record with `"trusted": false`;
  a `pass` is never used as proof of identity, and no decision (draft, compliance, fit) depends on it. Treat the
  lead's e-mail as unverified; the pipeline's own `contact_quality` check is separate.
* **The body is attacker-controlled text.** A deterministic heuristic flags instruction-like content addressed to
  an AI ("ignore previous instructions", "AI assistant: ...", role tags, requests to reveal keys, attempts to set the
  verdict, hidden HTML text). The mail's From display name, subject, and the extracted `name` / `company` /
  `job_title` are all scanned (these reach research and compliance prompts). A flagged mail gets status
  `needs_review`: **the pipeline is not run**, the LLM fill-in is skipped, no draft is written, and the flags are in
  the record. `hidden_html_text` alone is informational (marketing mail often carries hidden preheader text); any
  instruction-like text inside it fires the other heuristics. Extracted fields are also stripped of control
  characters and capped at 120 characters, and `prompt_safety.wrap_evidence` neutralises a literal `<<<` in wrapped
  text so it cannot close or forge an `EVIDENCE` block. The heuristic is regex-based and will miss paraphrases;
  it is a tripwire, not a classifier.
* The LLM fill-in only sees the mail wrapped as an `EVIDENCE` block with the standard injection warning, only for
  fields still empty, its answer is schema-validated, and a proposed website must be a plain `http(s)` URL whose host
  appears in the mail and is not a free-mail or shortener host.
* **Attachments are never decoded** into the pipeline, an LLM or the record; only filename, content type and size
  are kept. The whole message is capped (`LEADSCOUT_INBOUND_MAX_BYTES`).
* **Sending is off by default and gated separately from rep notifications.** A draft is a file in
  `out/inbound/drafts/`. It is sent only when ALL hold: `LEADSCOUT_SEND_EMAIL` and `LEADSCOUT_SEND_REPLIES` are on,
  SMTP credentials exist, `LEADSCOUT_PROOF_MODE` is off, the mail is not a forward, the reply goes to the From address
  itself (a Reply-To on another domain gets a draft only), and the TOPMOST `Authentication-Results` header shows
  `dmarc=pass` (or an aligned `dkim=pass`) for the From domain. That header is only as trustworthy as the hop that
  wrote it: unless your MTA overwrites inbound `Authentication-Results`, treat it as forgeable; webhook providers
  that add no such header can never reach the send path. The record's `draft.send_blocked_reason` names the failed
  condition. No draft is written for automatic/no-reply senders, injection-flagged mail, or compliance statuses
  outside the profile's `reply.draft_on` (default: `clear` only).
* **Message-ID squatting.** The Message-ID is chosen by the sender. Someone who learns the Message-ID of a mail that
  has not arrived yet can send a different mail with it first, and the real one is then recorded as a duplicate. The
  content-hash key does not help here. Accepted risk; the raw copies and `source`/`raw_sha256` in the record allow
  an audit.
* **Artefact names.** Outbox `.eml` and `results/*.json` for inbound mail are named by the record id, not by the
  mail-supplied company name, so one sender cannot overwrite another lead's files.
* **JS rendering.** `LEADSCOUT_JS_RENDER=1` makes `providers/website.py` hand URLs to a headless browser. The URL is
  checked with `assert_public_url` before navigation, but redirects the browser follows afterwards are NOT
  re-validated; keep JS render off for the inbound path (its URLs are attacker-supplied).
* **Access log.** `?token=` would appear in uvicorn's access log; `leadscout.api` installs a filter on
  `uvicorn.access` that redacts it. Proxies in front (nginx, Cloudflare) log their own copy: prefer the header, or
  redact there too.
* **Raw copies** in `out/inbound/raw/` are the full messages (personal data). Keep them under the same retention
  rules as the records; delete a raw copy once its record is terminal if you do not need it for audit.
* The webhook compares the token with `hmac.compare_digest`, is absent (404) until a token is configured, and
  reads at most ~1.5x the message cap. Use TLS; prefer the header over `?token=`.
* Records contain personal data (names, addresses, a body preview). Keep `out/inbound/` under the same retention
  rules as the tracker.
