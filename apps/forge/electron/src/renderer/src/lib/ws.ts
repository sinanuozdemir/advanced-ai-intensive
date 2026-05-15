// WebSocket helpers.
//
// `openChatSocket()` is the high-level wrapper the Chat view uses: it
// dispatches inbound events (typed) to a handler and auto-reconnects with a
// simple backoff. `openTraceSocket()` does the same for the "everything"
// channel that the Audit / Agents views observe.

import { apiBase } from './api';

export type ServerEvent =
  | { type: 'ws_hello'; ts: string; repo_root: string; channel?: string }
  | {
      type: 'permission_request';
      ts: string;
      request_id: string;
      tool: string;
      agent: string;
      reason: string;
      args: Record<string, unknown>;
      timeout_s: number;
    }
  | {
      type: 'permission_timeout';
      ts: string;
      request_id: string;
      tool: string;
      agent: string;
      approved: boolean;
    }
  | {
      type: 'chat_result';
      ts: string;
      thread_id: string;
      topology: string;
      planned: boolean;
      answer: string;
    }
  | { type: 'chat_error'; ts: string; error: string }
  | { type: 'agent_spawn'; ts: string; agent_name: string; kind: string }
  | { type: 'agent_done'; ts: string; agent_name: string; result?: string }
  | { type: 'tool_call'; ts: string; tool: string; agent_name: string; args?: Record<string, unknown> }
  | { type: 'tool_result'; ts: string; tool?: string; agent_name?: string; ok: boolean; preview?: string }
  | { type: 'memory_write'; ts: string }
  | { type: 'memory_read'; ts: string }
  | { type: 'compaction_fired'; ts: string; strategy: string }
  | { type: 'thread_end'; ts: string; task_id: string }
  | {
      type: 'thread_eval_ready';
      ts: string;
      task_id: string;
      outcome_overall: number | null;
      trajectory_overall: number | null;
      error?: string;
    }
  | { type: 'thread_eval_failed'; ts: string; task_id: string; error: string }
  | { type: 'thread_eval_skipped'; ts: string; task_id: string; reason: string }
  | { type: 'thread_eval_deleted'; ts: string; task_id: string; removed: number }
  | { type: 'thread_list_changed'; ts: string; thread_id: string }
  | {
      type: 'procedural_triggered';
      ts: string;
      task_id: string;
      judge_model: string;
      user_message_preview: string;
      skills: Array<{
        name: string;
        score: number;
        when_to_use: string;
        fragment: string;
        reasoning: string;
      }>;
    }
  | {
      type: 'model_in_use';
      ts: string;
      task_id: string;
      model: string;
      role: string;
      summarizer?: string;
      judge?: string;
    };

// NOTE: We deliberately don't include a catch-all
// ``{ type: string; [key: string]: unknown }`` member. The catch-all matched
// every string literal (a TS quirk with discriminated unions whose
// discriminant overlaps), which forced every named branch to widen back to
// ``unknown`` and broke narrowing in consumers like Chat.tsx. Unknown event
// types from the server are simply ignored by the typed handlers — and
// because the WS frame is parsed with ``JSON.parse(...) as ServerEvent``,
// a future event type would still come through; consumers just need to add
// a case for it here to get typed access.

export type ClientMessage =
  | { type: 'chat'; message: string; thread_id?: string; plan_mode?: boolean }
  | { type: 'permission_response'; request_id: string; approved: boolean }
  | { type: 'ping' };

export interface ForgeSocket {
  send: (msg: ClientMessage) => void;
  close: () => void;
  readonly readyState: () => number;
}

interface OpenSocketOpts {
  path: '/ws' | '/ws/chat';
  onEvent: (event: ServerEvent) => void;
  onOpen?: () => void;
  onClose?: () => void;
  onError?: (err: Event) => void;
}

function openSocket({
  path,
  onEvent,
  onOpen,
  onClose,
  onError,
}: OpenSocketOpts): ForgeSocket {
  const wsUrl =
    apiBase().replace(/^http/, 'ws') + path;

  let ws: WebSocket | null = null;
  let closed = false;
  let backoffMs = 500;
  const MAX_BACKOFF = 8_000;
  const pending: ClientMessage[] = [];

  const connect = () => {
    ws = new WebSocket(wsUrl);
    ws.onopen = () => {
      backoffMs = 500;
      // Flush anything that was queued while disconnected.
      while (pending.length > 0) {
        const msg = pending.shift();
        if (msg !== undefined && ws && ws.readyState === WebSocket.OPEN) {
          ws.send(JSON.stringify(msg));
        }
      }
      onOpen?.();
    };
    ws.onmessage = (e) => {
      try {
        const data = JSON.parse(e.data as string) as ServerEvent;
        onEvent(data);
      } catch {
        // Ignore malformed frames — the server never sends non-JSON.
      }
    };
    ws.onerror = (e) => {
      onError?.(e);
    };
    ws.onclose = () => {
      onClose?.();
      if (!closed) {
        // Exponential-ish backoff with a cap. The user shouldn't see
        // this except when `forge serve` is restarting.
        setTimeout(connect, backoffMs);
        backoffMs = Math.min(MAX_BACKOFF, Math.floor(backoffMs * 1.7));
      }
    };
  };
  connect();

  return {
    send(msg: ClientMessage) {
      if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify(msg));
      } else {
        pending.push(msg);
      }
    },
    close() {
      closed = true;
      ws?.close();
    },
    readyState: () => ws?.readyState ?? WebSocket.CLOSED,
  };
}

export function openChatSocket(opts: Omit<OpenSocketOpts, 'path'>): ForgeSocket {
  return openSocket({ ...opts, path: '/ws/chat' });
}

export function openTraceSocket(opts: Omit<OpenSocketOpts, 'path'>): ForgeSocket {
  return openSocket({ ...opts, path: '/ws' });
}
