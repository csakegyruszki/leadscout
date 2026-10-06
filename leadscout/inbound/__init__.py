"""Inbound e-mail entry point: a raw message in, a Lead through the pipeline, a record out.

See docs/inbound-email.md. Layers, bottom up: `parse` (RFC 822 -> ParsedMessage),
`extract` (ParsedMessage -> Lead fields, injection flags), `store` (idempotency),
`draft` (reply draft), `service` (orchestration), `imap` / `webhook` (transports).
"""
