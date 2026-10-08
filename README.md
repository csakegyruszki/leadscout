# leadscout
[![CI](https://github.com/csakegyruszki/leadscout/actions/workflows/ci.yml/badge.svg)](https://github.com/csakegyruszki/leadscout/actions/workflows/ci.yml)

Inbound lead triage as a Python service. A lead arrives — from a web form, the API, the CLI or an **incoming e-mail** — and leadscout researches the company, screens it for compliance, scores it against your ideal customer profile, and hands back a structured record with a clear next action. For mail, it can also prepare a reply draft.

```
Lead (form · API · CLI · e-mail)
  → Research      company website · Wikipedia/Wikidata · GLEIF · public network evidence · LLM summary
  → Compliance    competitor match · restricted jurisdictions · sanctions screening → clear / review / blocked
  → Fit score     0-100, deterministic, weights from your profile
  → Output        JSON record · Excel tracker row · rep notification · optional reply draft
```

Every fact behind a decision is linked to a captured source in a provenance ledger. Nothing is sent by default.

## Quick start

```bash
pip install -e .
cp .env.example .env          # one LLM endpoint is enough

leadscout lead --name "Dana Whitfield" --email dana@example.com \
  --company "Example Ltd" --website example.com --size-band 201-1000
leadscout batch samples/leads.json
leadscout mail process message.eml --json        # add --no-pipeline for a dry run
uvicorn leadscout.api:app                        # web form + webhook on :8000
```

Any LLM API works: OpenRouter, Cloudflare Workers AI, Ollama (local, no key), Anthropic, Gemini, or any OpenAI-compatible endpoint. Set the chain with `LEADSCOUT_MODELS`; a failing hop falls through to the next. Sanctions screening, company-size lookup and e-mail verification keys are optional.

## Configuration profiles

One YAML file per deployment under `config/profiles/`, selected with `LEADSCOUT_ORG_PROFILE` (name or path):

- `identity`: product, sender, the rep who is notified, signature
- `fit`: threshold, size bands, scoring weights and rules
- `compliance`: competitor list, restricted jurisdictions, what a competitor sells
- `reply`, `notification`, `research`: draft template and policy, subject line, research focus

The `default` profile is neutral B2B; an example profile for an infrastructure vendor sits next to it. Environment variables override the profile where both exist.

## Wiring to your inbox

Forwarded leads, enquiries and replies all work: the parser unwraps forwards, strips quoted history and signatures, and ignores attachments. Each message is processed once (Message-ID plus content hash). Mail containing instructions aimed at an AI is held for review instead of being processed.

- **IMAP**: set `LEADSCOUT_IMAP_HOST`, `_USER`, `_PASSWORD` (an app password), run `leadscout mail poll --once`, then enable the systemd timer in `infra/systemd/`.
- **Webhook**: set `LEADSCOUT_INBOUND_TOKEN` and point your provider at `POST /inbound/mail` (Cloudflare Email Routing via an Email Worker, Mailgun, SendGrid Inbound Parse or Postmark).

Output: `out/inbound/records/<id>.json` and, when the profile allows it, a draft in `out/inbound/drafts/`. Sending requires `LEADSCOUT_SEND_EMAIL=1`, `LEADSCOUT_SEND_REPLIES=1`, SMTP settings and an authenticated sender. Recipes and security notes: [docs/inbound-email.md](docs/inbound-email.md).

## Development

```bash
make test     # offline test suite
make lint
make eval     # compliance evals
```

A Docker image (`docker build .`) serves the same API.

Screening is a support tool, not a legal or KYC determination. OpenSanctions data is [CC BY-NC 4.0](https://www.opensanctions.org/licensing/).

© 2026 Nikita Khava. See [LICENSE](LICENSE).
