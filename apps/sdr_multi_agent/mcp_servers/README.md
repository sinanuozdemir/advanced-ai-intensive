# MCP Servers

Model Context Protocol servers used by the SDR multi-agent app. The Flask app (and the supervisor) connect to these via stdio.

## Servers

### `hubspot_server/` — HubSpot CRM

Custom MCP server for HubSpot CRM read/write. Tools:

- `create_contact` — create a new contact (standard fields + lifecycle stage).
- `fetch_contacts` — search/filter contacts (query, company, job title, lifecycle, email domain, limit 1-100).
- `update_contact` — update any contact fields, lifecycle stage, lead status, notes.
- `attach_note_to_contact` — create and attach a note to a contact.
- `retrieve_all_notes_for_contact` — fetch all notes for a contact.

Auth: `HUBSPOT_API_KEY` (HubSpot Private App token with `crm.objects.contacts.read` and `crm.objects.contacts.write`).

### `research_server/` — Web research

Tools:

- `web_search` — top-3 SerpApi results (title, snippet, URL).
- `scrape_website` — Firecrawl page fetch, markdown or link-extraction modes.

Auth: `SERPAPI_API_KEY`, `FIRECRAWL_API_KEY`.

### `email_server/` — Resend email send

Vendored fork of [`resend/mcp-send-email`](https://github.com/resend/mcp-send-email). Tool:

- `send-email` — sends through Resend (`RESEND_API_KEY`, `SENDER_EMAIL_ADDRESS`, optional `REPLY_TO_EMAIL_ADDRESSES`).

See `email_server/README.md` for details.

## Running with Docker

All servers are wired into the app's Compose file. From `apps/sdr_multi_agent/`:

```bash
docker compose up --build
```

To start only one MCP server:

```bash
docker compose up --build hubspot-mcp-server
docker compose up --build research-mcp-server
docker compose up --build email-mcp-server
```

## Environment

API keys are read from the host environment (and `.env`) by Compose:

```yaml
environment:
  - HUBSPOT_API_KEY=${HUBSPOT_API_KEY:-...}
```

Set them via `.env` (see `apps/sdr_multi_agent/.env.example`) or export them in your shell before `docker compose up`.

## Endpoints

- Flask UI: http://localhost:8080 (host port 8080 → container 5000).
- MCP servers: stdio only (no HTTP).
- Postgres: localhost:5432.
- RabbitMQ management UI: http://localhost:15672.

## HubSpot key — how to get one

1. HubSpot account → Settings → Integrations → Private Apps.
2. Create a private app.
3. Grant `crm.objects.contacts.read` and `crm.objects.contacts.write`. Add `crm.objects.notes.read`/`crm.objects.notes.write` if you want the note tools.
4. Copy the access token and set `HUBSPOT_API_KEY`.
