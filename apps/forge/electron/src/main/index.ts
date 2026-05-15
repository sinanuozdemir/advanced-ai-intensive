// Electron main process for the Forge desktop app.
//
// Responsibilities (in order):
//   1. Find the repo root (walk up from cwd looking for .git or .forge).
//   2. Read the server port from <repo>/.forge/config.toml (regex; one int).
//   3. Probe http://127.0.0.1:<port>/api/health. If unreachable, spawn
//      `forge serve` as a child process and wait for it to come up.
//   4. Open a single BrowserWindow that loads the bundled renderer (Vite
//      build output in production, dev server in development).
//
// Node integration is OFF in the renderer; everything the renderer needs from
// the system goes through the preload's `window.forgeApi`.

import { app, BrowserWindow, ipcMain, shell } from 'electron';
import { spawn, type ChildProcess } from 'node:child_process';
import * as http from 'node:http';
import * as path from 'node:path';
import * as fs from 'node:fs';

const DEFAULT_PORT = 6790;
const SERVER_HOST = '127.0.0.1';

let serverProc: ChildProcess | null = null;
let serverPort: number = DEFAULT_PORT;

function findRepoRoot(): string {
  let cur = process.cwd();
  const root = path.parse(cur).root;
  while (cur !== root) {
    if (
      fs.existsSync(path.join(cur, '.git')) ||
      fs.existsSync(path.join(cur, '.forge'))
    ) {
      return cur;
    }
    cur = path.dirname(cur);
  }
  // Fall back to apps/forge/electron's grandparent (the monorepo root) when
  // launched from inside the electron dir itself.
  return path.resolve(__dirname, '..', '..', '..', '..');
}

function readPortFromConfig(repo: string): number {
  try {
    const toml = fs.readFileSync(
      path.join(repo, '.forge', 'config.toml'),
      'utf8',
    );
    const m = toml.match(/server_port\s*=\s*(\d+)/);
    if (m) return parseInt(m[1], 10);
  } catch {
    // ignore — fall through to default
  }
  return DEFAULT_PORT;
}

function probe(host: string, port: number): Promise<boolean> {
  return new Promise((resolve) => {
    const req = http.get(
      { host, port, path: '/api/health', timeout: 800 },
      (res) => {
        let body = '';
        res.on('data', (c) => (body += c));
        res.on('end', () => resolve(res.statusCode === 200));
      },
    );
    req.on('error', () => resolve(false));
    req.on('timeout', () => {
      req.destroy();
      resolve(false);
    });
  });
}

async function waitForServer(
  host: string,
  port: number,
  maxMs = 15_000,
): Promise<boolean> {
  const t0 = Date.now();
  while (Date.now() - t0 < maxMs) {
    if (await probe(host, port)) return true;
    await new Promise((r) => setTimeout(r, 250));
  }
  return false;
}

async function ensureBackend(repo: string): Promise<void> {
  serverPort = readPortFromConfig(repo);
  if (await probe(SERVER_HOST, serverPort)) {
    console.log(`[forge-ui] reusing existing server on :${serverPort}`);
    return;
  }
  console.log(`[forge-ui] starting \`forge serve\` on :${serverPort}`);
  const env = { ...process.env, FORGE_REPO: repo };
  serverProc = spawn(
    'forge',
    ['--repo', repo, 'serve', '--port', String(serverPort)],
    { env, stdio: ['ignore', 'pipe', 'pipe'] },
  );
  serverProc.stdout?.on('data', (d) =>
    process.stdout.write(`[forge serve] ${d}`),
  );
  serverProc.stderr?.on('data', (d) =>
    process.stderr.write(`[forge serve ERR] ${d}`),
  );
  serverProc.on('exit', (code) => {
    console.log(`[forge-ui] server exited (code=${code})`);
    serverProc = null;
  });
  const up = await waitForServer(SERVER_HOST, serverPort);
  if (!up) {
    console.error(
      '[forge-ui] server did not come up within 15s. Is `forge` on PATH?',
    );
  }
}

function createWindow(): void {
  const win = new BrowserWindow({
    width: 1280,
    height: 800,
    minWidth: 880,
    minHeight: 540,
    backgroundColor: '#0b0d12',
    title: 'Forge',
    webPreferences: {
      preload: path.join(__dirname, '..', 'preload', 'index.js'),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true,
    },
  });

  // In dev, electron-vite exposes the renderer on a local Vite server and
  // sets ELECTRON_RENDERER_URL. In prod, load the file:// build output.
  const devUrl = process.env['ELECTRON_RENDERER_URL'];
  const apiBase = `http://${SERVER_HOST}:${serverPort}`;
  if (devUrl) {
    win.loadURL(`${devUrl}?api=${encodeURIComponent(apiBase)}`);
  } else {
    win.loadFile(path.join(__dirname, '..', 'renderer', 'index.html'), {
      search: `api=${encodeURIComponent(apiBase)}`,
    });
  }
  if (process.env['ELECTRON_OPEN_DEVTOOLS']) {
    win.webContents.openDevTools({ mode: 'detach' });
  }
}

ipcMain.handle('forge:open-external', async (_evt, url: unknown) => {
  if (typeof url === 'string' && /^https?:\/\//.test(url)) {
    await shell.openExternal(url);
  }
});

app.whenReady().then(async () => {
  const repo = findRepoRoot();
  console.log(`[forge-ui] repo = ${repo}`);
  await ensureBackend(repo);
  createWindow();
  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
  });
});

app.on('window-all-closed', () => {
  if (serverProc) {
    try {
      serverProc.kill('SIGTERM');
    } catch {
      // ignore
    }
  }
  if (process.platform !== 'darwin') app.quit();
});
