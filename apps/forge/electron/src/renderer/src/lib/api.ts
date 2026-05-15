// Thin REST client around the Forge backend.
//
// The renderer reads the API base URL from the `?api=...` query string that
// the main process appended to the renderer URL. In dev the URL also comes
// from the query string (set by electron.vite via the main process).

const params = new URLSearchParams(window.location.search);
const API_BASE = params.get('api') || 'http://127.0.0.1:6790';

export function apiBase(): string {
  return API_BASE;
}

export class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message);
    this.name = 'ApiError';
  }
}

async function request<T>(
  path: string,
  init?: RequestInit & { json?: unknown },
): Promise<T> {
  const headers: Record<string, string> = {
    Accept: 'application/json',
    ...(init?.headers as Record<string, string> | undefined),
  };
  let body = init?.body;
  if (init?.json !== undefined) {
    headers['Content-Type'] = 'application/json';
    body = JSON.stringify(init.json);
  }
  const res = await fetch(`${API_BASE}${path}`, { ...init, headers, body });
  if (!res.ok) {
    const text = await res.text().catch(() => '');
    throw new ApiError(res.status, text || res.statusText);
  }
  const ct = res.headers.get('content-type') || '';
  if (ct.includes('application/json')) return (await res.json()) as T;
  return (await res.text()) as unknown as T;
}

// ---------------------------------------------------------- endpoint types --

export interface WorkspaceInfo {
  repo_root: string;
  branch: string | null;
  head: string | null;
  dirty: boolean;
  is_git: boolean;
}

export interface HealthInfo {
  ok: boolean;
  repo_root: string;
  version: string;
  ts: string;
  engine_started: boolean;
}

export interface ChatPayload {
  message: string;
  thread_id?: string;
  plan_mode?: boolean;
}

export interface ChatResult {
  thread_id: string;
  topology: string;
  planned: boolean;
  answer: string;
}

export interface ModelHealth {
  ok: boolean;
  provider: 'openrouter' | 'ollama' | string;
  slug: string;
  latency_ms: number;
  error?: string;
}

export interface ModelCatalog {
  openrouter: string[];
  openrouter_error: string | null;
  ollama: string[];
  ollama_host: string;
  ollama_available: boolean;
  ollama_error: string | null;
}

// ------------------------------------------------------------------- API ----

export const api = {
  health: () => request<HealthInfo>('/api/health'),
  workspace: () => request<WorkspaceInfo>('/api/workspace'),
  config: () => request<Record<string, unknown>>('/api/config'),
  configSchema: () => request<Record<string, unknown>>('/api/config/schema'),
  putConfig: (cfg: Record<string, unknown>) =>
    request<{ ok: boolean; config: Record<string, unknown> }>('/api/config', {
      method: 'PUT',
      json: cfg,
    }),
  modelHealth: (slug: string) =>
    request<ModelHealth>(
      `/api/models/health?slug=${encodeURIComponent(slug)}`,
    ),
  modelCatalog: () => request<ModelCatalog>('/api/models/catalog'),
  agents: () => request<Array<Record<string, unknown>>>('/api/agents'),
  putAgent: (name: string, body: Record<string, unknown>) =>
    request<Record<string, unknown>>(`/api/agents/${encodeURIComponent(name)}`, {
      method: 'PUT',
      json: body,
    }),
  deleteAgent: (name: string) =>
    request<{ ok: boolean }>(`/api/agents/${encodeURIComponent(name)}`, {
      method: 'DELETE',
    }),
  tools: () =>
    request<{
      servers: Array<{
        name: string;
        tools: Array<{ name: string; description: string }>;
      }>;
    }>('/api/tools'),
  semanticSearch: (q: string, k = 10) =>
    request<
      Array<{
        id: string;
        text: string;
        score: number;
        thread_id?: string;
        created_at?: string;
      }>
    >(`/api/memory/semantic?q=${encodeURIComponent(q)}&k=${k}`),
  semanticBrowse: (k = 50) =>
    request<
      Array<{
        id: string;
        text: string;
        score: number;
        thread_id?: string;
        created_at?: string;
      }>
    >(`/api/memory/semantic?k=${k}`),
  deleteSemantic: (id: string) =>
    request<{ ok: boolean; id: string }>(
      `/api/memory/semantic/${encodeURIComponent(id)}`,
      { method: 'DELETE' },
    ),
  episodic: (limit = 20) =>
    request<
      Array<{
        summary: string;
        thread_id: string;
        score: number;
        created_at: string;
        updated_at?: string;
        id: string;
      }>
    >(`/api/memory/episodic?limit=${limit}`),
  procedural: () =>
    request<Array<{ name: string; fragment: string; when_to_use: string; usage_count: number; score: number; created_at: string }>>(
      '/api/memory/procedural',
    ),
  chat: (payload: ChatPayload) =>
    request<ChatResult>('/api/chat', { method: 'POST', json: payload }),
  trace: (tail = 200) =>
    request<string>(`/api/trace?tail=${tail}`),
  audit: (tail = 200) =>
    request<string>(`/api/audit?tail=${tail}`),
  // Chat threads (recall + resume)
  listThreads: (limit = 50) =>
    request<{ threads: ThreadSummary[]; count: number }>(
      `/api/threads?limit=${limit}`,
    ),
  getThread: (threadId: string) =>
    request<{ thread_id: string; turns: TranscriptTurn[] }>(
      `/api/threads/${encodeURIComponent(threadId)}`,
    ),
};

// Thread sidebar / resume types
export interface ThreadSummary {
  thread_id: string;
  title: string;
  first_ts: string;
  last_ts: string;
  turns: number;
  last_answer: string;
  ok: boolean;
}

export type TraceEntryWire =
  | { kind: 'policy'; mode: string; topology: string; reason: string }
  | { kind: 'plan'; head: string }
  | { kind: 'spawn'; name: string; agentKind: string }
  | { kind: 'agent_done'; name: string }
  | {
      kind: 'tool';
      name: string;
      agent: string;
      args: Record<string, unknown>;
      status: 'pending' | 'ok' | 'error';
      preview?: string;
    }
  | { kind: 'compaction'; strategy: string }
  | {
      kind: 'procedural';
      skills: Array<{
        name: string;
        score: number;
        when_to_use: string;
        fragment: string;
        reasoning: string;
      }>;
      judgeModel: string;
    }
  | {
      kind: 'model';
      model: string;
      role: string;
      summarizer?: string;
      judge?: string;
    };

export interface TranscriptTurn {
  user: string;
  assistant: string;
  ok: boolean;
  error: string;
  trace: TraceEntryWire[];
}
