import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  openChatSocket,
  type ForgeSocket,
  type ServerEvent,
} from '@/lib/ws';
import { api, type ThreadSummary, type TraceEntryWire } from '@/lib/api';
import { cn } from '@/lib/cn';
import {
  Send,
  CheckCircle2,
  AlertCircle,
  Loader2,
  ChevronDown,
  ChevronRight,
  Wrench,
  MessageSquarePlus,
  History,
  Sparkles,
  Brain,
  PanelRightClose,
  PanelRightOpen,
} from 'lucide-react';

// LocalStorage key for "auto-focus the last thread I was in".
const LAST_THREAD_KEY = 'forge.chat.lastThreadId';

// ---------------------------------------------------------------- types -----

interface PermissionAsk {
  request_id: string;
  tool: string;
  agent: string;
  reason: string;
  args: Record<string, unknown>;
  timeout_s: number;
  receivedAt: number;
}

/**
 * Inline trace entries shown between user input and the assistant answer.
 *
 * ``tool`` entries are mutable: they enter as ``status: 'pending'`` when a
 * ``tool_call`` arrives and flip to ``ok`` / ``error`` when the matching
 * ``tool_result`` lands. Pairing is FIFO (Forge's solo agents call tools
 * serially; the supervisor occasionally interleaves but tracer order is
 * still per-agent, so first-pending-with-matching-name is reliable enough
 * for a live view).
 */
type TraceEntry =
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

/**
 * One reflection session — everything between an ``agent_spawn`` carrying
 * ``agent_name="reflector"`` and the matching ``agent_done``. Rendered in
 * the right-docked side panel (not in the chat bubble) so the user can
 * see the main agent's answer immediately while reflection runs in the
 * background.
 */
interface ReflectionEntry {
  kind: 'tool';
  name: string;
  args: Record<string, unknown>;
  status: 'pending' | 'ok' | 'error';
  preview?: string;
}
interface ReflectionSession {
  /** Unique id (we use the WS ``ts`` of the spawn event) so React keys
   *  are stable even when two sessions for the same thread coexist. */
  id: string;
  thread_id: string;
  started_at: string;
  finished_at?: string;
  /** Short result string from ``agent_done`` (e.g. "summary=yes skills=1"). */
  result?: string;
  entries: ReflectionEntry[];
}

interface ChatMessage {
  role: 'user' | 'assistant' | 'system';
  text: string;
  /** True while the assistant turn is still streaming. The bubble shows
   *  trace cards + a spinner; ``text`` fills in when ``chat_result`` lands. */
  streaming?: boolean;
  /** Trace events for this turn, in arrival order. */
  trace?: TraceEntry[];
  thread_id?: string;
}

// ---------------------------------------------------------------- view ------

/**
 * Mutate the last (streaming) assistant message. No-op if there isn't one,
 * which happens for trace events that arrive before the user has sent
 * anything. No-op for trace events that arrive before the user has sent
 * a message.
 */
function patchStreaming(
  msgs: ChatMessage[],
  patch: (m: ChatMessage) => ChatMessage,
): ChatMessage[] {
  const idx = msgs.length - 1;
  if (idx < 0) return msgs;
  const last = msgs[idx];
  if (last.role !== 'assistant' || !last.streaming) return msgs;
  const next = msgs.slice();
  next[idx] = patch(last);
  return next;
}

function pushTrace(m: ChatMessage, entry: TraceEntry): ChatMessage {
  return { ...m, trace: [...(m.trace ?? []), entry] };
}

function settleLastPendingTool(
  m: ChatMessage,
  toolName: string,
  ok: boolean,
  preview: string,
): ChatMessage {
  const trace = (m.trace ?? []).slice();
  // Walk backwards to the most recent pending tool with the matching name.
  for (let i = trace.length - 1; i >= 0; i--) {
    const t = trace[i];
    if (t.kind === 'tool' && t.status === 'pending' && t.name === toolName) {
      trace[i] = {
        ...t,
        status: ok ? 'ok' : 'error',
        preview,
      };
      return { ...m, trace };
    }
  }
  // If we never saw the tool_call (e.g. the user reloaded mid-turn), still
  // record the result so it's visible.
  trace.push({
    kind: 'tool',
    name: toolName,
    agent: '',
    args: {},
    status: ok ? 'ok' : 'error',
    preview,
  });
  return { ...m, trace };
}

// How many reflection sessions to keep in the side panel before the
// oldest get evicted. The reflector typically runs once per turn, so 12
// covers ~12 turns of history without growing unbounded.
const MAX_REFLECTION_SESSIONS = 12;

export function ChatView(): JSX.Element {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState('');
  const [threadId, setThreadId] = useState<string | undefined>(undefined);
  const [ask, setAsk] = useState<PermissionAsk | null>(null);
  const [connected, setConnected] = useState(false);
  const [threadList, setThreadList] = useState<ThreadSummary[]>([]);
  const [loadingThread, setLoadingThread] = useState(false);
  // Reflection panel: sessions go newest-first. ``open`` is persisted so
  // re-opening the app remembers the user's preference.
  const [reflectionSessions, setReflectionSessions] = useState<ReflectionSession[]>([]);
  const [reflectionPanelOpen, setReflectionPanelOpen] = useState<boolean>(() => {
    try {
      const stored = window.localStorage.getItem('forge.chat.reflectionPanelOpen');
      return stored === null ? true : stored === '1';
    } catch {
      return true;
    }
  });
  useEffect(() => {
    try {
      window.localStorage.setItem(
        'forge.chat.reflectionPanelOpen', reflectionPanelOpen ? '1' : '0',
      );
    } catch {
      /* ignore */
    }
  }, [reflectionPanelOpen]);
  const sockRef = useRef<ForgeSocket | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);
  // Keep ``threadId`` accessible inside ``handleEvent`` without
  // re-creating the callback every time the user switches threads
  // (which would force the WS to re-open).
  const threadIdRef = useRef<string | undefined>(undefined);
  useEffect(() => {
    threadIdRef.current = threadId;
  }, [threadId]);

  // ``true`` while at least one reflection session is still in flight —
  // used by the side-panel header to show a small spinner.
  const reflectionRunning = useMemo(
    () => reflectionSessions.some((s) => !s.finished_at),
    [reflectionSessions],
  );

  // Track which thread we've already hydrated into the view so the
  // ``thread_list_changed`` handler doesn't auto-jump us back to the top
  // after every turn we just sent. Only ``null`` means "Chat just
  // mounted, please pick the most recent thread for me."
  const hydratedThreadRef = useRef<string | null>(null);

  // Derived: are we waiting on a turn? True iff the latest message is a
  // streaming assistant placeholder. Keeps state minimal — no second source
  // of truth that could drift.
  const running = useMemo(() => {
    const last = messages[messages.length - 1];
    return Boolean(last && last.role === 'assistant' && last.streaming);
  }, [messages]);

  // -------- thread sidebar: list, load-one, new-chat -------------------

  const refreshThreadList = useCallback(async () => {
    try {
      const r = await api.listThreads(50);
      setThreadList(r.threads);
      return r.threads;
    } catch {
      // Sidebar is non-critical — failure here shouldn't block typing.
      return [] as ThreadSummary[];
    }
  }, []);

  const loadThread = useCallback(
    async (tid: string) => {
      setLoadingThread(true);
      try {
        const r = await api.getThread(tid);
        setMessages(transcriptToMessages(r.turns));
        setThreadId(tid);
        hydratedThreadRef.current = tid;
        try {
          window.localStorage.setItem(LAST_THREAD_KEY, tid);
        } catch {
          /* localStorage disabled — non-fatal */
        }
      } catch {
        // Thread missing / 404. Fall back to fresh chat so the user
        // can keep typing.
        setMessages([]);
        setThreadId(undefined);
        hydratedThreadRef.current = null;
      } finally {
        setLoadingThread(false);
      }
    },
    [],
  );

  const newChat = useCallback(() => {
    setMessages([]);
    setThreadId(undefined);
    hydratedThreadRef.current = null;
    setInput('');
    try {
      window.localStorage.removeItem(LAST_THREAD_KEY);
    } catch {
      /* ignore */
    }
  }, []);

  const handleEvent = useCallback((ev: ServerEvent) => {
    if (ev.type === 'ws_hello') return;

    if (ev.type === 'permission_request') {
      setAsk({
        request_id: ev.request_id,
        tool: ev.tool,
        agent: ev.agent,
        reason: ev.reason,
        args: ev.args,
        timeout_s: ev.timeout_s,
        receivedAt: Date.now(),
      });
      return;
    }
    if (ev.type === 'permission_timeout') {
      setAsk((cur) => (cur?.request_id === ev.request_id ? null : cur));
      return;
    }
    if (ev.type === 'chat_error') {
      setMessages((m) =>
        patchStreaming(m, (cur) => ({
          ...cur,
          streaming: false,
          text: cur.text || `error: ${ev.error}`,
          role: 'system',
        })),
      );
      return;
    }
    if (ev.type === 'chat_result') {
      setThreadId(ev.thread_id);
      hydratedThreadRef.current = ev.thread_id;
      try {
        window.localStorage.setItem(LAST_THREAD_KEY, ev.thread_id);
      } catch {
        /* ignore */
      }
      setMessages((m) =>
        patchStreaming(m, (cur) => ({
          ...cur,
          streaming: false,
          text: ev.answer,
          thread_id: ev.thread_id,
        })),
      );
      return;
    }
    if (ev.type === 'thread_list_changed') {
      // Refresh the sidebar in place; never re-hydrate the current view
      // from this signal — the user is mid-conversation, jumping them
      // around would be worse than a slightly stale list.
      void refreshThreadList();
      return;
    }

    // ---- streaming trace events: pushed into the current assistant turn ----
    //
    // Reflector activity (anything where ``agent_name === "reflector"``)
    // is routed into the side ReflectionPanel instead of the chat bubble.
    // Reflection runs asynchronously after the main agent finishes, so
    // these events typically arrive AFTER ``chat_result`` (i.e. when the
    // bubble is no longer streaming) — pushing them into the chat would
    // either silently drop them or contaminate a later turn.
    switch (ev.type) {
      case 'agent_spawn':
        if (ev.agent_name === 'reflector') {
          setReflectionSessions((prev) => {
            const next: ReflectionSession = {
              id: ev.ts,
              thread_id: threadIdRef.current ?? '',
              started_at: ev.ts,
              entries: [],
            };
            const merged = [next, ...prev];
            return merged.slice(0, MAX_REFLECTION_SESSIONS);
          });
          return;
        }
        setMessages((m) =>
          patchStreaming(m, (cur) =>
            pushTrace(cur, {
              kind: 'spawn',
              name: String(ev.agent_name ?? ''),
              agentKind: String(ev.kind ?? ''),
            }),
          ),
        );
        return;
      case 'agent_done':
        if (ev.agent_name === 'reflector') {
          setReflectionSessions((prev) => {
            if (prev.length === 0) return prev;
            // Settle the most-recent (head) session — that's the one
            // that just finished, since sessions are stored newest-first.
            const [head, ...rest] = prev;
            return [
              {
                ...head,
                finished_at: ev.ts,
                result: ev.result ? String(ev.result) : 'done',
              },
              ...rest,
            ];
          });
          return;
        }
        setMessages((m) =>
          patchStreaming(m, (cur) =>
            pushTrace(cur, {
              kind: 'agent_done',
              name: String(ev.agent_name ?? ''),
            }),
          ),
        );
        return;
      case 'tool_call':
        if (ev.agent_name === 'reflector') {
          setReflectionSessions((prev) => {
            if (prev.length === 0) return prev;
            const [head, ...rest] = prev;
            return [
              {
                ...head,
                entries: [
                  ...head.entries,
                  {
                    kind: 'tool',
                    name: String(ev.tool ?? ''),
                    args: (ev.args as Record<string, unknown>) ?? {},
                    status: 'pending',
                  },
                ],
              },
              ...rest,
            ];
          });
          return;
        }
        setMessages((m) =>
          patchStreaming(m, (cur) =>
            pushTrace(cur, {
              kind: 'tool',
              name: String(ev.tool ?? ''),
              agent: String(ev.agent_name ?? ''),
              args: (ev.args as Record<string, unknown>) ?? {},
              status: 'pending',
            }),
          ),
        );
        return;
      case 'tool_result':
        if (ev.agent_name === 'reflector') {
          setReflectionSessions((prev) => {
            if (prev.length === 0) return prev;
            const [head, ...rest] = prev;
            const toolName = String(ev.tool ?? '');
            const ok = ev.ok !== false;
            const preview = String(ev.preview ?? '');
            const entries = head.entries.slice();
            // Walk backwards to settle the most-recent pending entry
            // with the matching name — mirrors ``settleLastPendingTool``.
            for (let i = entries.length - 1; i >= 0; i--) {
              const t = entries[i];
              if (t.status === 'pending' && t.name === toolName) {
                entries[i] = {
                  ...t,
                  status: ok ? 'ok' : 'error',
                  preview,
                };
                return [{ ...head, entries }, ...rest];
              }
            }
            // No matching pending entry — append a settled one so it's
            // still visible (e.g. user reloaded mid-reflection).
            entries.push({
              kind: 'tool',
              name: toolName,
              args: {},
              status: ok ? 'ok' : 'error',
              preview,
            });
            return [{ ...head, entries }, ...rest];
          });
          return;
        }
        setMessages((m) =>
          patchStreaming(m, (cur) =>
            settleLastPendingTool(
              cur,
              String(ev.tool ?? ''),
              ev.ok !== false,
              String(ev.preview ?? ''),
            ),
          ),
        );
        return;
      case 'compaction_fired':
        setMessages((m) =>
          patchStreaming(m, (cur) =>
            pushTrace(cur, {
              kind: 'compaction',
              strategy: String(ev.strategy ?? ''),
            }),
          ),
        );
        return;
      case 'procedural_triggered':
        setMessages((m) =>
          patchStreaming(m, (cur) =>
            pushTrace(cur, {
              kind: 'procedural',
              skills: ev.skills ?? [],
              judgeModel: String(ev.judge_model ?? ''),
            }),
          ),
        );
        return;
      case 'model_in_use':
        setMessages((m) =>
          patchStreaming(m, (cur) =>
            pushTrace(cur, {
              kind: 'model',
              model: String(ev.model ?? ''),
              role: String(ev.role ?? ''),
              summarizer: ev.summarizer ? String(ev.summarizer) : undefined,
              judge: ev.judge ? String(ev.judge) : undefined,
            }),
          ),
        );
        return;
      default:
        return;
    }
  }, [refreshThreadList]);

  // Open WS once.
  useEffect(() => {
    const sock = openChatSocket({
      onEvent: handleEvent,
      onOpen: () => setConnected(true),
      onClose: () => setConnected(false),
    });
    sockRef.current = sock;
    return () => sock.close();
  }, [handleEvent]);

  // On mount: fetch threads + auto-load whichever is "most relevant"
  // (the last one the user was looking at, or the most recent if there's
  // no localStorage hint). The user explicitly asked for this: switching
  // away from Chat and coming back should restore where you were.
  useEffect(() => {
    let cancelled = false;
    (async () => {
      const threads = await refreshThreadList();
      if (cancelled || hydratedThreadRef.current !== null) return;
      if (threads.length === 0) return;
      let target: string | null = null;
      try {
        const last = window.localStorage.getItem(LAST_THREAD_KEY);
        if (last && threads.some((t) => t.thread_id === last)) {
          target = last;
        }
      } catch {
        /* ignore */
      }
      if (!target) target = threads[0].thread_id;
      await loadThread(target);
    })();
    return () => {
      cancelled = true;
    };
  }, [refreshThreadList, loadThread]);

  // Auto-scroll to the bottom on new messages.
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages, running, ask]);

  const send = useCallback(() => {
    const text = input.trim();
    if (!text || running || !sockRef.current) return;
    // Append both the user message AND an empty streaming assistant bubble
    // so trace events have somewhere to land as they arrive.
    setMessages((m) => [
      ...m,
      { role: 'user', text },
      { role: 'assistant', text: '', streaming: true, trace: [] },
    ]);
    setInput('');
    sockRef.current.send({ type: 'chat', message: text, thread_id: threadId });
  }, [input, running, threadId]);

  const respondPermission = useCallback(
    (approved: boolean) => {
      if (!ask || !sockRef.current) return;
      sockRef.current.send({
        type: 'permission_response',
        request_id: ask.request_id,
        approved,
      });
      setAsk(null);
    },
    [ask],
  );

  const placeholder = useMemo(() => {
    if (!connected) return 'connecting…';
    if (running) return 'forge is thinking…';
    return 'message forge';
  }, [connected, running]);

  return (
    <div className="relative flex h-full">
      <ThreadSidebar
        threads={threadList}
        activeId={threadId}
        loading={loadingThread}
        onSelect={(tid) => {
          if (tid !== threadId) void loadThread(tid);
        }}
        onNew={newChat}
      />
      <div className="relative flex h-full min-w-0 flex-1 flex-col">
      <div ref={scrollRef} className="flex-1 overflow-y-auto px-6 py-4">
        {messages.length === 0 && (
          <div className="mt-10 text-center text-sm text-muted-foreground">
            <p>start a conversation</p>
            <p className="mt-1 opacity-70">
              tool calls + agent spawns will appear inline
            </p>
          </div>
        )}
        <div className="mx-auto flex max-w-3xl flex-col gap-4">
          {messages.map((m, i) => (
            <MessageBubble key={i} msg={m} />
          ))}
        </div>
      </div>

      {/* Composer */}
      <div className="border-t border-border bg-card px-4 py-3">
        <div className="mx-auto flex max-w-3xl items-end gap-2">
          <textarea
            className="flex-1 resize-none rounded-md border border-input bg-background px-3 py-2 text-sm outline-none ring-ring focus:ring-2"
            rows={2}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                send();
              }
            }}
            placeholder={placeholder}
            disabled={running}
          />
          <button
            type="button"
            onClick={send}
            disabled={running || !input.trim()}
            className={cn(
              'flex h-10 w-10 items-center justify-center rounded-md',
              'bg-primary text-primary-foreground transition-opacity',
              'disabled:cursor-not-allowed disabled:opacity-40',
            )}
          >
            <Send className="h-4 w-4" />
          </button>
        </div>
      </div>

      {/* Permission modal */}
      {ask && (
        <PermissionModal
          ask={ask}
          onAllow={() => respondPermission(true)}
          onDeny={() => respondPermission(false)}
        />
      )}
      </div>
      <ReflectionPanel
        sessions={reflectionSessions}
        running={reflectionRunning}
        open={reflectionPanelOpen}
        onToggle={() => setReflectionPanelOpen((v) => !v)}
        onClear={() => setReflectionSessions([])}
      />
    </div>
  );
}

// --------------------------------------------------------- thread sidebar --

/**
 * Left-rail list of past chat threads with a "new chat" button at the
 * top. Sorted newest-first by the backend so the user's most recent
 * conversation is always the first row. Click to resume; the parent
 * fetches the transcript and rehydrates the right pane.
 */
function ThreadSidebar({
  threads,
  activeId,
  loading,
  onSelect,
  onNew,
}: {
  threads: ThreadSummary[];
  activeId: string | undefined;
  loading: boolean;
  onSelect: (tid: string) => void;
  onNew: () => void;
}): JSX.Element {
  return (
    <aside className="flex h-full w-60 shrink-0 flex-col border-r border-border bg-card/40">
      <div className="flex items-center justify-between border-b border-border px-3 py-2">
        <div className="flex items-center gap-2 text-[10px] font-semibold uppercase tracking-wide text-muted-foreground">
          <History className="h-3 w-3" />
          threads
        </div>
        <button
          type="button"
          onClick={onNew}
          className="flex items-center gap-1 rounded-md px-2 py-1 text-xs hover:bg-accent"
          title="start a new chat"
        >
          <MessageSquarePlus className="h-3.5 w-3.5" />
          new
        </button>
      </div>
      <div className="flex-1 overflow-y-auto">
        {threads.length === 0 ? (
          <div className="px-3 py-4 text-xs text-muted-foreground">
            no past conversations yet — say hi below to start one.
          </div>
        ) : (
          <ul className="flex flex-col">
            {threads.map((t) => {
              const isActive = activeId === t.thread_id;
              return (
                <li key={t.thread_id}>
                  <button
                    type="button"
                    onClick={() => onSelect(t.thread_id)}
                    className={cn(
                      'flex w-full flex-col items-start gap-0.5 border-b border-border/40 px-3 py-2 text-left text-xs hover:bg-accent/40',
                      isActive && 'bg-accent/60',
                    )}
                  >
                    <div className="line-clamp-2 w-full font-medium">
                      {t.title || '(empty prompt)'}
                    </div>
                    <div className="flex w-full items-center justify-between text-[10px] text-muted-foreground">
                      <span>
                        {t.turns} turn{t.turns === 1 ? '' : 's'}
                      </span>
                      <span>{relativeTime(t.last_ts)}</span>
                    </div>
                  </button>
                </li>
              );
            })}
          </ul>
        )}
      </div>
      {loading && (
        <div className="flex items-center gap-2 border-t border-border px-3 py-1.5 text-[10px] text-muted-foreground">
          <Loader2 className="h-3 w-3 animate-spin" /> loading transcript…
        </div>
      )}
    </aside>
  );
}

function relativeTime(iso: string | null | undefined): string {
  if (!iso) return '';
  const t = new Date(iso).getTime();
  if (!Number.isFinite(t)) return '';
  const dt = Date.now() - t;
  if (dt < 60_000) return 'just now';
  if (dt < 3600_000) return `${Math.floor(dt / 60_000)}m ago`;
  if (dt < 86_400_000) return `${Math.floor(dt / 3600_000)}h ago`;
  return `${Math.floor(dt / 86_400_000)}d ago`;
}

// ---------------------------------------------------- transcript hydration -

/** Turn the API's ``turns[]`` into the in-memory ``ChatMessage[]`` the
 *  view renders. Each turn becomes one user bubble + one assistant
 *  bubble; the assistant bubble carries the same trace shape used live,
 *  so the UI can't tell history from a live turn. */
function transcriptToMessages(
  turns: Array<{
    user: string;
    assistant: string;
    ok: boolean;
    error: string;
    trace: TraceEntryWire[];
  }>,
): ChatMessage[] {
  const out: ChatMessage[] = [];
  for (const t of turns) {
    out.push({ role: 'user', text: t.user });
    out.push({
      role: 'assistant',
      text: t.assistant || (t.ok ? '' : `error: ${t.error}`),
      trace: t.trace as TraceEntry[],
      streaming: false,
    });
  }
  return out;
}

// ---------------------------------------------------------------- bubble ----

function MessageBubble({ msg }: { msg: ChatMessage }): JSX.Element {
  return (
    <div
      className={cn(
        'rounded-lg border px-4 py-3 text-sm',
        msg.role === 'user'
          ? 'border-primary/30 bg-primary/5 ml-12 self-end'
          : msg.role === 'assistant'
            ? 'border-border bg-card mr-12 self-start'
            : 'border-destructive/30 bg-destructive/5 text-destructive',
      )}
    >
      <div className="mb-1 flex items-center gap-2 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
        <span>{msg.role}</span>
        {msg.streaming && (
          <Loader2 className="h-3 w-3 animate-spin text-primary" />
        )}
      </div>
      {msg.trace && msg.trace.length > 0 && (
        <div className="mb-2 flex flex-col gap-1.5">
          {msg.trace.map((t, i) => (
            <TraceLine key={i} entry={t} />
          ))}
        </div>
      )}
      {msg.text && (
        <div className="whitespace-pre-wrap leading-relaxed">{msg.text}</div>
      )}
    </div>
  );
}

/** Render one trace entry. Tool entries get an expandable card with args +
 *  result preview; procedural entries get a card with skill details;
 *  everything else is a single-line pill. */
function TraceLine({ entry }: { entry: TraceEntry }): JSX.Element {
  if (entry.kind === 'tool') {
    return <ToolCard entry={entry} />;
  }
  if (entry.kind === 'procedural') {
    return <ProceduralCard entry={entry} />;
  }
  return <TraceBadge entry={entry} />;
}

function TraceBadge({
  entry,
}: {
  entry: Exclude<TraceEntry, { kind: 'tool' | 'procedural' }>;
}): JSX.Element {
  switch (entry.kind) {
    case 'model': {
      const sub = [
        entry.summarizer ? `summarizer: ${entry.summarizer}` : '',
        entry.judge ? `judge: ${entry.judge}` : '',
      ]
        .filter(Boolean)
        .join(' · ');
      return (
        <span
          className="self-start rounded-sm bg-cyan-500/15 px-1.5 py-0.5 font-mono text-[10px] text-cyan-300"
          title={sub || undefined}
        >
          model: {entry.model}
        </span>
      );
    }
    case 'policy':
      return (
        <span className="self-start rounded-sm bg-secondary px-1.5 py-0.5 text-[10px] uppercase tracking-wide text-secondary-foreground">
          {entry.mode}/{entry.topology}
        </span>
      );
    case 'plan':
      return (
        <span
          className="self-start rounded-sm bg-amber-500/15 px-1.5 py-0.5 text-[10px] text-amber-300"
          title={entry.head}
        >
          plan: {entry.head}
        </span>
      );
    case 'spawn':
      return (
        <span className="self-start rounded-sm bg-emerald-500/15 px-1.5 py-0.5 text-[10px] text-emerald-300">
          spawned {entry.name}
          {entry.agentKind ? ` (${entry.agentKind})` : ''}
        </span>
      );
    case 'agent_done':
      return (
        <span className="self-start rounded-sm bg-blue-500/15 px-1.5 py-0.5 text-[10px] text-blue-300">
          {entry.name} done
        </span>
      );
    case 'compaction':
      return (
        <span className="self-start rounded-sm bg-purple-500/15 px-1.5 py-0.5 text-[10px] text-purple-300">
          compact: {entry.strategy}
        </span>
      );
    default:
      return <span />;
  }
}

function ToolCard({
  entry,
}: {
  entry: Extract<TraceEntry, { kind: 'tool' }>;
}): JSX.Element {
  const [open, setOpen] = useState(false);
  const statusIcon =
    entry.status === 'pending' ? (
      <Loader2 className="h-3 w-3 animate-spin text-amber-300" />
    ) : entry.status === 'ok' ? (
      <CheckCircle2 className="h-3 w-3 text-emerald-400" />
    ) : (
      <AlertCircle className="h-3 w-3 text-destructive" />
    );
  const argsPreview = useMemo(() => oneLineArgs(entry.args), [entry.args]);

  return (
    <div className="rounded-md border border-border/60 bg-background/40">
      <button
        type="button"
        onClick={() => setOpen((x) => !x)}
        className="flex w-full items-center gap-2 px-2.5 py-1.5 text-left text-xs hover:bg-accent/40"
      >
        {open ? (
          <ChevronDown className="h-3 w-3 shrink-0 text-muted-foreground" />
        ) : (
          <ChevronRight className="h-3 w-3 shrink-0 text-muted-foreground" />
        )}
        <Wrench className="h-3 w-3 shrink-0 text-muted-foreground" />
        <span className="font-mono font-medium">{entry.name}</span>
        {entry.agent && entry.agent !== 'main' && (
          <span className="rounded-sm bg-accent px-1 py-px text-[10px] text-muted-foreground">
            {entry.agent}
          </span>
        )}
        <span className="ml-1 flex-1 truncate text-muted-foreground">
          {argsPreview}
        </span>
        {statusIcon}
      </button>
      {open && (
        <div className="space-y-2 border-t border-border/60 px-2.5 py-2 text-xs">
          <section>
            <div className="mb-1 text-[10px] uppercase tracking-wide text-muted-foreground">
              args
            </div>
            <pre className="max-h-48 overflow-auto rounded bg-background/60 p-2 font-mono text-[11px] leading-snug">
              {Object.keys(entry.args).length === 0
                ? '(none)'
                : JSON.stringify(entry.args, null, 2)}
            </pre>
          </section>
          {entry.preview !== undefined && (
            <section>
              <div className="mb-1 text-[10px] uppercase tracking-wide text-muted-foreground">
                result {entry.status === 'error' ? '(error)' : 'preview'}
              </div>
              <pre
                className={cn(
                  'max-h-48 overflow-auto rounded bg-background/60 p-2 font-mono text-[11px] leading-snug whitespace-pre-wrap',
                  entry.status === 'error' && 'text-destructive',
                )}
              >
                {entry.preview || '(empty)'}
              </pre>
            </section>
          )}
        </div>
      )}
    </div>
  );
}

/** Render the procedural-skill recall card. Collapsed view shows a small
 *  sparkles chip with the kept skill names; expanding it reveals each
 *  skill's fragment, when_to_use cue, judge reasoning, and cosine score. */
function ProceduralCard({
  entry,
}: {
  entry: Extract<TraceEntry, { kind: 'procedural' }>;
}): JSX.Element {
  const [open, setOpen] = useState(false);
  if (!entry.skills.length) return <span />;
  const header =
    entry.skills.length === 1
      ? entry.skills[0].name
      : `${entry.skills.length} skills`;
  return (
    <div className="rounded-md border border-violet-500/30 bg-violet-500/5">
      <button
        type="button"
        onClick={() => setOpen((x) => !x)}
        className="flex w-full items-center gap-2 px-2.5 py-1.5 text-left text-xs hover:bg-violet-500/10"
      >
        {open ? (
          <ChevronDown className="h-3 w-3 shrink-0 text-violet-300" />
        ) : (
          <ChevronRight className="h-3 w-3 shrink-0 text-violet-300" />
        )}
        <Sparkles className="h-3 w-3 shrink-0 text-violet-300" />
        <span className="font-medium text-violet-200">
          procedural recall: {header}
        </span>
        {entry.judgeModel && (
          <span className="ml-1 flex-1 truncate text-[10px] text-muted-foreground">
            judged by {entry.judgeModel}
          </span>
        )}
      </button>
      {open && (
        <div className="space-y-3 border-t border-violet-500/30 px-2.5 py-2 text-xs">
          {entry.skills.map((s) => (
            <section key={s.name} className="space-y-1">
              <div className="flex items-baseline justify-between gap-2">
                <span className="font-mono font-medium text-violet-200">
                  {s.name}
                </span>
                <span className="text-[10px] text-muted-foreground">
                  cosine {s.score.toFixed(3)}
                </span>
              </div>
              {s.when_to_use && (
                <div className="text-[11px] italic text-muted-foreground">
                  when: {s.when_to_use}
                </div>
              )}
              {s.reasoning && (
                <div className="rounded bg-violet-500/10 px-2 py-1 text-[11px] text-violet-100/90">
                  <span className="font-medium text-violet-300">
                    why kept:
                  </span>{' '}
                  {s.reasoning}
                </div>
              )}
              <p className="whitespace-pre-wrap leading-relaxed text-foreground/90">
                {s.fragment}
              </p>
            </section>
          ))}
        </div>
      )}
    </div>
  );
}

/** Render a tool's args as a compact one-line summary for the collapsed
 *  card header. Falls back to a JSON snippet for unfamiliar shapes. */
function oneLineArgs(args: Record<string, unknown>): string {
  const keys = Object.keys(args);
  if (keys.length === 0) return '';
  // Common case: one or two scalar args (path, query, command, ...). Render
  // ``key=value, key=value`` truncated. Avoids dumping huge JSON into the
  // header when the user just wants a glance.
  const parts: string[] = [];
  for (const k of keys) {
    const v = args[k];
    const rendered =
      typeof v === 'string'
        ? v
        : typeof v === 'number' || typeof v === 'boolean'
          ? String(v)
          : JSON.stringify(v);
    parts.push(`${k}=${rendered}`);
  }
  const line = parts.join(', ');
  return line.length > 100 ? line.slice(0, 97) + '…' : line;
}

// ------------------------------------------------------ reflection panel ---

/**
 * Right-docked panel showing the reflector agent's activity asynchronously.
 *
 * Reflection runs in the background after every chat turn (engine emits
 * ``thread_end`` BEFORE scheduling the reflector). The panel collects
 * ``agent_spawn`` / ``tool_call`` / ``tool_result`` / ``agent_done``
 * events that carry ``agent_name === "reflector"`` and groups them into
 * collapsible session cards (newest first), so the user can keep reading
 * the main agent's answer while reflection happens.
 *
 * The whole panel collapses to a slim ~10px gutter when ``open`` is
 * ``false``; that's persisted in localStorage so the user's choice
 * survives reloads.
 */
function ReflectionPanel({
  sessions,
  running,
  open,
  onToggle,
  onClear,
}: {
  sessions: ReflectionSession[];
  running: boolean;
  open: boolean;
  onToggle: () => void;
  onClear: () => void;
}): JSX.Element {
  if (!open) {
    return (
      <aside className="flex h-full w-8 shrink-0 flex-col items-center justify-start border-l border-border bg-card/30 pt-3">
        <button
          type="button"
          onClick={onToggle}
          className="flex flex-col items-center gap-1 rounded-md px-1 py-2 text-muted-foreground hover:bg-accent hover:text-foreground"
          title="show reflection panel"
        >
          <PanelRightOpen className="h-4 w-4" />
          {running && (
            <Loader2 className="h-3 w-3 animate-spin text-primary" />
          )}
        </button>
      </aside>
    );
  }
  return (
    <aside className="flex h-full w-80 shrink-0 flex-col border-l border-border bg-card/40">
      <div className="flex items-center justify-between border-b border-border px-3 py-2">
        <div className="flex items-center gap-2 text-[10px] font-semibold uppercase tracking-wide text-muted-foreground">
          <Brain className="h-3 w-3" />
          reflection
          {running && (
            <Loader2 className="h-3 w-3 animate-spin text-primary" />
          )}
        </div>
        <div className="flex items-center gap-1">
          {sessions.length > 0 && (
            <button
              type="button"
              onClick={onClear}
              className="rounded-md px-1.5 py-0.5 text-[10px] text-muted-foreground hover:bg-accent hover:text-foreground"
              title="clear reflection log"
            >
              clear
            </button>
          )}
          <button
            type="button"
            onClick={onToggle}
            className="rounded-md p-1 text-muted-foreground hover:bg-accent hover:text-foreground"
            title="hide reflection panel"
          >
            <PanelRightClose className="h-3.5 w-3.5" />
          </button>
        </div>
      </div>
      <div className="flex-1 overflow-y-auto px-2 py-2">
        {sessions.length === 0 ? (
          <p className="px-2 py-6 text-center text-[11px] text-muted-foreground">
            reflection runs in the background after each turn — saved
            episodes and any new skills will appear here without
            blocking the chat.
          </p>
        ) : (
          <ul className="flex flex-col gap-2">
            {sessions.map((s) => (
              <li key={s.id}>
                <ReflectionSessionCard session={s} />
              </li>
            ))}
          </ul>
        )}
      </div>
    </aside>
  );
}

function ReflectionSessionCard({
  session,
}: {
  session: ReflectionSession;
}): JSX.Element {
  const [open, setOpen] = useState<boolean>(!session.finished_at);
  const inFlight = !session.finished_at;
  // ``thread_id`` may be empty when we lost the link (e.g. after reload);
  // hide the suffix in that case so we don't render an ugly trailing dot.
  const tidShort = session.thread_id ? session.thread_id.slice(0, 8) : '';
  return (
    <section className="rounded-md border border-border bg-background/40">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="flex w-full items-center justify-between gap-2 px-2 py-1.5 text-left text-[11px] hover:bg-accent/50"
      >
        <div className="flex min-w-0 items-center gap-1.5">
          {inFlight ? (
            <Loader2 className="h-3 w-3 shrink-0 animate-spin text-primary" />
          ) : (
            <CheckCircle2 className="h-3 w-3 shrink-0 text-emerald-400" />
          )}
          <span className="font-medium">
            {inFlight ? 'reflecting…' : 'reflection done'}
          </span>
          {tidShort && (
            <span className="truncate text-muted-foreground">· {tidShort}</span>
          )}
        </div>
        {open ? (
          <ChevronDown className="h-3 w-3 shrink-0 text-muted-foreground" />
        ) : (
          <ChevronRight className="h-3 w-3 shrink-0 text-muted-foreground" />
        )}
      </button>
      {open && (
        <div className="border-t border-border/60 px-2 py-1.5">
          {session.result && (
            <div className="mb-1.5 text-[10px] text-muted-foreground">
              <span className="font-medium text-foreground/80">result:</span>{' '}
              <span className="font-mono">{session.result}</span>
            </div>
          )}
          {session.entries.length === 0 ? (
            <p className="text-[11px] italic text-muted-foreground">
              waiting on first tool call…
            </p>
          ) : (
            <ul className="flex flex-col gap-1">
              {session.entries.map((e, i) => (
                <li
                  key={i}
                  className="flex items-start gap-1.5 text-[11px]"
                >
                  {e.status === 'pending' ? (
                    <Loader2 className="mt-0.5 h-3 w-3 shrink-0 animate-spin text-primary" />
                  ) : e.status === 'ok' ? (
                    <CheckCircle2 className="mt-0.5 h-3 w-3 shrink-0 text-emerald-400" />
                  ) : (
                    <AlertCircle className="mt-0.5 h-3 w-3 shrink-0 text-destructive" />
                  )}
                  <div className="min-w-0 flex-1">
                    <div className="flex items-center gap-1.5">
                      <Wrench className="h-2.5 w-2.5 shrink-0 text-muted-foreground" />
                      <span className="font-mono">{e.name}</span>
                    </div>
                    {Object.keys(e.args).length > 0 && (
                      <div className="truncate font-mono text-[10px] text-muted-foreground">
                        {oneLineArgs(e.args)}
                      </div>
                    )}
                    {e.preview && (
                      <div
                        className={cn(
                          'mt-0.5 truncate text-[10px]',
                          e.status === 'error'
                            ? 'text-destructive'
                            : 'text-muted-foreground',
                        )}
                        title={e.preview}
                      >
                        {e.preview}
                      </div>
                    )}
                  </div>
                </li>
              ))}
            </ul>
          )}
        </div>
      )}
    </section>
  );
}

// --------------------------------------------------------- permission modal -

function PermissionModal({
  ask,
  onAllow,
  onDeny,
}: {
  ask: PermissionAsk;
  onAllow: () => void;
  onDeny: () => void;
}): JSX.Element {
  return (
    <div className="absolute inset-0 z-50 flex items-center justify-center bg-background/80 backdrop-blur-sm">
      <div className="w-[480px] max-w-[90vw] rounded-lg border border-amber-500/50 bg-card p-5 shadow-xl">
        <div className="mb-2 flex items-center gap-2 text-amber-300">
          <span className="h-2 w-2 rounded-full bg-amber-400" />
          <h2 className="text-sm font-semibold uppercase tracking-wide">
            permission requested
          </h2>
        </div>
        <dl className="space-y-1 text-sm">
          <div className="flex gap-2">
            <dt className="w-16 text-muted-foreground">tool</dt>
            <dd className="font-mono">{ask.tool}</dd>
          </div>
          <div className="flex gap-2">
            <dt className="w-16 text-muted-foreground">agent</dt>
            <dd>{ask.agent}</dd>
          </div>
          <div className="flex gap-2">
            <dt className="w-16 text-muted-foreground">reason</dt>
            <dd className="text-muted-foreground">{ask.reason}</dd>
          </div>
        </dl>
        <pre className="mt-3 max-h-48 overflow-auto rounded-md bg-background/60 p-3 text-xs">
          {JSON.stringify(ask.args, null, 2)}
        </pre>
        <p className="mt-2 text-xs text-muted-foreground">
          auto-approving in {ask.timeout_s}s if no response
        </p>
        <div className="mt-4 flex justify-end gap-2">
          <button
            type="button"
            onClick={onDeny}
            className="rounded-md border border-border px-3 py-1.5 text-sm hover:bg-accent"
          >
            deny
          </button>
          <button
            type="button"
            onClick={onAllow}
            className="rounded-md bg-primary px-3 py-1.5 text-sm font-medium text-primary-foreground"
          >
            allow
          </button>
        </div>
      </div>
    </div>
  );
}
