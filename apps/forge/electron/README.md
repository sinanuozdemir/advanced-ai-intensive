# Forge desktop app

Vite + React + TypeScript + Tailwind shell that talks to `forge serve` over
loopback.

## Tech

- **electron-vite** — three build targets (main, preload, renderer) with HMR
  for the renderer in dev.
- **React 18 + TypeScript** — strict mode, `@/*` alias to `src/renderer/src/`.
- **Tailwind CSS** — shadcn-compatible CSS variables (`--primary`, `--background`,
  etc.) so any shadcn component drops in cleanly via
  `npx shadcn@latest add ...`.
- **lucide-react** — icons.

## Layout

```
apps/forge/electron/
├── electron.vite.config.ts          ← build config (main / preload / renderer)
├── package.json                     ← deps + electron-builder packaging
├── postcss.config.js
├── tailwind.config.ts
├── tsconfig.json / tsconfig.node.json
└── src/
    ├── main/index.ts                ← Electron main: finds repo, spawns `forge serve`
    ├── preload/index.ts             ← tiny bridge (openExternal)
    └── renderer/
        ├── index.html
        └── src/
            ├── App.tsx              ← left rail + workspace header + view router
            ├── main.tsx             ← React entry
            ├── index.css            ← Tailwind layers + dark theme tokens
            ├── components/
            │   ├── Sidebar.tsx
            │   └── WorkspaceHeader.tsx
            ├── lib/
            │   ├── api.ts           ← REST client (typed per endpoint)
            │   ├── cn.ts            ← clsx + tailwind-merge
            │   └── ws.ts            ← chat + trace WebSocket helpers
            └── views/
                ├── Chat.tsx         ← WS-streamed chat + inline tool/agent/plan cards + permission modal
                ├── Agents.tsx       ← CRUD for persistent agents (JSON editor)
                ├── Memory.tsx       ← semantic / episodic / procedural tabs
                ├── Mcp.tsx          ← drag-drop MCP install + server list
                └── Settings.tsx     ← schema-driven form against /api/config/schema
```

## Develop

```bash
cd apps/forge/electron
npm install
npm run dev          # electron-vite dev server + auto-launch Electron, HMR on the renderer
```

## Build / package

```bash
npm run build        # bundle into out/ (main, preload, renderer)
npm run dist         # electron-builder produces a dmg/AppImage/exe in release/
```

## How it talks to the backend

The main process probes `127.0.0.1:<port>/api/health`. If unreachable it
spawns `forge serve` as a child process and waits for the health endpoint
to come up before opening the BrowserWindow. The renderer reads the API
base URL out of the `?api=...` query string the main process appends, then
uses plain `fetch` and `WebSocket` — no IPC for data flow.

The chat view opens **one** WebSocket against `/ws/chat`. Outbound messages:
`chat` (start a turn) and `permission_response` (answer a permission ask).
Inbound: the full server event stream (trace, agent spawns, tool calls,
permission asks, chat result). Streaming-chat with REST fallback would only
kick in if the WS upgrade fails — we haven't needed it.
