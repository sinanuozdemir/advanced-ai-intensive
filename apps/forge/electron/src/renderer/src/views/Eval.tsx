import { useCallback, useEffect, useMemo, useState } from 'react';
import {
  Gauge,
  RefreshCcw,
  Loader2,
  ChevronDown,
  ChevronRight,
  ChevronLeft,
  Wrench,
  Info,
  AlertCircle,
  Trash2,
} from 'lucide-react';
import { apiBase } from '@/lib/api';
import { cn } from '@/lib/cn';
import { openTraceSocket, type ForgeSocket } from '@/lib/ws';

// Eval tab. Lists per-thread auto-evals (outcome rubric + trajectory rubric,
// both LLM-as-judge) and lets the user drill into one to see the rubric
// breakdown plus the trajectory the judge saw. Re-run is one click.

interface ToolCall {
  agent: string;
  tool: string;
  args: Record<string, unknown>;
  ok: boolean | null;
  preview: string;
}

interface RubricScore {
  overall?: number;
  rationale?: string;
  [k: string]: unknown;
}

interface ThreadEval {
  thread_id: string;
  user_task: string;
  final_answer: string;
  topology: string;
  tool_calls: ToolCall[];
  outcome: RubricScore;
  trajectory: RubricScore;
  judge_models: { outcome: string; trajectory: string };
  ts: string;
  elapsed_s: number;
  error: string;
  // Optional because older JSONL rows written before per-turn slicing
  // landed don't carry these fields. Renderers fall back to 1/1.
  turn_index?: number;
  turn_count?: number;
}

interface RubricsResp {
  prompts: { outcome: string; trajectory: string };
  config: {
    auto_evaluate_threads: boolean;
    outcome_judge_model: string;
    trajectory_judge_model: string;
  };
}

export function EvalView(): JSX.Element {
  const [evals, setEvals] = useState<ThreadEval[]>([]);
  // Selection is keyed on (thread_id, ts) because a thread can have
  // multiple eval rows (one per turn). The old single-string ``thread_id``
  // selector was the source of the "I can only see the most recent turn"
  // bug: ``evals.find(e => e.thread_id === id)`` always returned the
  // newest row, regardless of which sidebar entry the user clicked.
  const [selectedTs, setSelectedTs] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [rubricInfo, setRubricInfo] = useState<RubricsResp | null>(null);
  const [showPrompts, setShowPrompts] = useState(false);

  const refresh = useCallback(async () => {
    setLoading(true);
    try {
      const r = await fetch(`${apiBase()}/api/eval/threads?limit=200`);
      if (!r.ok) throw new Error(`${r.status}`);
      const data = (await r.json()) as { evals: ThreadEval[] };
      setEvals(data.evals ?? []);
      setErr(null);
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }, []);

  // First-load: fetch evals + rubric prompts in parallel.
  useEffect(() => {
    void refresh();
    void (async () => {
      try {
        const r = await fetch(`${apiBase()}/api/eval/rubrics`);
        if (r.ok) setRubricInfo((await r.json()) as RubricsResp);
      } catch {
        // Best-effort; the prompts panel just won't render.
      }
    })();
  }, [refresh]);

  // Live refresh: when a new eval lands (or gets deleted), the engine
  // emits an event on the trace WS. We don't try to merge in just one row
  // — easier to refetch the whole list, which is cheap.
  useEffect(() => {
    const sock: ForgeSocket = openTraceSocket({
      onEvent: (ev) => {
        if (
          ev.type === 'thread_eval_ready' ||
          ev.type === 'thread_eval_failed' ||
          ev.type === 'thread_eval_deleted'
        ) {
          void refresh();
        }
      },
    });
    return () => sock.close();
  }, [refresh]);

  const deleteEval = useCallback(
    async (threadId: string) => {
      try {
        const r = await fetch(
          `${apiBase()}/api/eval/threads/${encodeURIComponent(threadId)}`,
          { method: 'DELETE' },
        );
        if (!r.ok) throw new Error(await r.text());
        // Drop the selection if it pointed at any turn of this thread.
        setSelectedTs((cur) => {
          if (!cur) return cur;
          const row = evals.find((e) => e.ts === cur);
          return row && row.thread_id === threadId ? null : cur;
        });
        await refresh();
      } catch (e) {
        setErr(e instanceof Error ? e.message : String(e));
      }
    },
    [refresh, evals],
  );

  const selected = useMemo(
    () => evals.find((e) => e.ts === selectedTs) ?? null,
    [evals, selectedTs],
  );

  // All evals for the currently-selected thread, oldest-first so the
  // "Turn N of M" navigator reads chronologically.
  const turnsForSelected = useMemo(() => {
    if (!selected) return [] as ThreadEval[];
    return evals
      .filter((e) => e.thread_id === selected.thread_id)
      .slice()
      .sort((a, b) => (a.ts < b.ts ? -1 : a.ts > b.ts ? 1 : 0));
  }, [evals, selected]);

  const stats = useMemo(() => {
    if (evals.length === 0) {
      return { n: 0, outcome: null as number | null, trajectory: null as number | null };
    }
    const outcomes = evals
      .map((e) => e.outcome?.overall)
      .filter((x): x is number => typeof x === 'number');
    const trajectories = evals
      .map((e) => e.trajectory?.overall)
      .filter((x): x is number => typeof x === 'number');
    const avg = (xs: number[]) =>
      xs.length === 0 ? null : xs.reduce((a, b) => a + b, 0) / xs.length;
    return {
      n: evals.length,
      outcome: avg(outcomes),
      trajectory: avg(trajectories),
    };
  }, [evals]);

  return (
    <div className="flex h-full flex-col">
      <header className="flex items-center justify-between gap-4 border-b border-border bg-card px-4 py-2 text-xs">
        <div className="flex items-center gap-2 font-semibold uppercase tracking-wide text-muted-foreground">
          <Gauge className="h-4 w-4" />
          <span>Thread evals</span>
        </div>
        <div className="flex items-center gap-4 text-muted-foreground">
          <span>
            {stats.n} eval{stats.n === 1 ? '' : 's'}
          </span>
          <Stat label="outcome" value={stats.outcome} />
          <Stat label="trajectory" value={stats.trajectory} />
          <button
            type="button"
            onClick={() => setShowPrompts((x) => !x)}
            className="flex items-center gap-1 rounded-md border border-border px-2 py-1 hover:bg-accent"
          >
            <Info className="h-3 w-3" /> rubric
          </button>
          <button
            type="button"
            onClick={refresh}
            disabled={loading}
            className="rounded p-1 hover:bg-accent disabled:opacity-40"
            title="refresh"
          >
            {loading ? (
              <Loader2 className="h-3 w-3 animate-spin" />
            ) : (
              <RefreshCcw className="h-3 w-3" />
            )}
          </button>
        </div>
      </header>

      {showPrompts && rubricInfo && (
        <RubricPromptsPanel info={rubricInfo} onClose={() => setShowPrompts(false)} />
      )}

      <div className="grid flex-1 grid-cols-[24rem_1fr] overflow-hidden">
        <aside className="flex flex-col overflow-y-auto border-r border-border">
          {err && (
            <div className="m-3 rounded-md bg-destructive/10 p-2 text-xs text-destructive">
              {err}
            </div>
          )}
          {evals.length === 0 && !loading && (
            <div className="m-6 text-center text-xs text-muted-foreground">
              no evals yet — send a message in chat and the auto-eval will
              appear here.
            </div>
          )}
          <ul className="flex flex-col">
            {evals.map((e) => (
              <li
                key={e.thread_id + e.ts}
                className={cn(
                  'group relative border-b border-border/60',
                  selectedTs === e.ts && 'bg-accent/60',
                )}
              >
                <button
                  type="button"
                  onClick={() => setSelectedTs(e.ts)}
                  className="flex w-full flex-col items-start gap-1 px-3 py-2 pr-9 text-left text-xs hover:bg-accent/40"
                >
                  <div className="flex w-full items-center justify-between gap-2">
                    <span className="truncate font-medium">
                      {e.user_task || '(no task)'}
                    </span>
                    <ScorePill
                      label="O"
                      value={e.outcome?.overall}
                      title="outcome"
                    />
                    <ScorePill
                      label="T"
                      value={e.trajectory?.overall}
                      title="trajectory"
                    />
                  </div>
                  <div className="flex w-full items-center justify-between text-[10px] text-muted-foreground">
                    <span className="truncate font-mono">{e.thread_id}</span>
                    <div className="flex items-center gap-2">
                      {(e.turn_count ?? 1) > 1 && (
                        <span
                          className="rounded-sm bg-secondary px-1 py-px text-secondary-foreground"
                          title={`turn ${e.turn_index ?? 1} of ${e.turn_count ?? 1}`}
                        >
                          t{e.turn_index ?? 1}/{e.turn_count ?? 1}
                        </span>
                      )}
                      <span>{relative(e.ts)}</span>
                    </div>
                  </div>
                  {e.error && (
                    <div className="flex items-center gap-1 text-[10px] text-amber-400">
                      <AlertCircle className="h-2.5 w-2.5" /> {e.error.slice(0, 80)}
                    </div>
                  )}
                </button>
                <button
                  type="button"
                  onClick={(ev) => {
                    ev.stopPropagation();
                    if (
                      window.confirm(
                        `Delete eval(s) for thread ${e.thread_id}?\nThis removes every stored eval row for this thread.`,
                      )
                    ) {
                      void deleteEval(e.thread_id);
                    }
                  }}
                  className="absolute right-2 top-2 rounded p-1 text-muted-foreground opacity-0 transition-opacity hover:bg-destructive/15 hover:text-destructive group-hover:opacity-100"
                  title="delete this eval"
                >
                  <Trash2 className="h-3 w-3" />
                </button>
              </li>
            ))}
          </ul>
        </aside>

        <section className="flex flex-col overflow-y-auto">
          {selected ? (
            <EvalDetail
              eval={selected}
              turns={turnsForSelected}
              onPickTurn={(ts) => setSelectedTs(ts)}
              onRerun={refresh}
              onDelete={() => deleteEval(selected.thread_id)}
            />
          ) : (
            <div className="m-10 text-center text-sm text-muted-foreground">
              pick a thread on the left to see its rubric scores + the
              trajectory the judge saw.
            </div>
          )}
        </section>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------- subviews ---

function Stat({
  label,
  value,
}: {
  label: string;
  value: number | null;
}): JSX.Element {
  return (
    <span>
      <span className="text-[10px] uppercase tracking-wide">{label}</span>{' '}
      <span className="font-mono text-foreground">
        {value === null ? '—' : value.toFixed(2)}
      </span>
    </span>
  );
}

function ScorePill({
  label,
  value,
  title,
}: {
  label: string;
  value: number | undefined;
  title: string;
}): JSX.Element {
  const v = typeof value === 'number' ? value : null;
  return (
    <span
      title={`${title}: ${v === null ? 'n/a' : v.toFixed(2)} / 5`}
      className={cn(
        'shrink-0 rounded-sm px-1.5 py-0.5 font-mono text-[10px] font-semibold',
        v === null
          ? 'bg-muted text-muted-foreground'
          : v >= 4
            ? 'bg-emerald-500/20 text-emerald-300'
            : v >= 2.5
              ? 'bg-amber-500/15 text-amber-300'
              : 'bg-destructive/15 text-destructive',
      )}
    >
      {label}:{v === null ? '—' : v.toFixed(1)}
    </span>
  );
}

function RubricPromptsPanel({
  info,
  onClose,
}: {
  info: RubricsResp;
  onClose: () => void;
}): JSX.Element {
  return (
    <section className="border-b border-border bg-card/60 px-4 py-3 text-xs">
      <div className="mb-2 flex items-center justify-between text-muted-foreground">
        <span className="font-semibold uppercase tracking-wide">
          how the judge scores
        </span>
        <div className="flex items-center gap-3">
          <span>outcome model: <span className="font-mono">{info.config.outcome_judge_model}</span></span>
          <span>trajectory model: <span className="font-mono">{info.config.trajectory_judge_model}</span></span>
          <button
            type="button"
            onClick={onClose}
            className="rounded-md border border-border px-2 py-0.5 hover:bg-accent"
          >
            hide
          </button>
        </div>
      </div>
      <div className="grid grid-cols-2 gap-3">
        <div>
          <div className="mb-1 font-semibold">outcome rubric</div>
          <pre className="max-h-44 overflow-auto whitespace-pre-wrap rounded bg-background/60 p-2 text-[11px] leading-snug">
            {info.prompts.outcome}
          </pre>
        </div>
        <div>
          <div className="mb-1 font-semibold">trajectory rubric</div>
          <pre className="max-h-44 overflow-auto whitespace-pre-wrap rounded bg-background/60 p-2 text-[11px] leading-snug">
            {info.prompts.trajectory}
          </pre>
        </div>
      </div>
    </section>
  );
}

function EvalDetail({
  eval: e,
  turns,
  onPickTurn,
  onRerun,
  onDelete,
}: {
  eval: ThreadEval;
  turns: ThreadEval[];
  onPickTurn: (ts: string) => void;
  onRerun: () => void | Promise<void>;
  onDelete: () => void | Promise<void>;
}): JSX.Element {
  const [rerunning, setRerunning] = useState(false);
  const [rerunErr, setRerunErr] = useState<string | null>(null);

  // Position of the current eval inside the thread (oldest-first list).
  // Fall back to 1/1 when older rows are missing turn_index — the
  // sidebar list still serves as a per-row navigator in that case.
  const turnPosition = Math.max(0, turns.findIndex((t) => t.ts === e.ts));
  const turnCount = turns.length;
  const prevTurn = turnPosition > 0 ? turns[turnPosition - 1] : null;
  const nextTurn = turnPosition < turnCount - 1 ? turns[turnPosition + 1] : null;

  const rerun = async () => {
    setRerunning(true);
    setRerunErr(null);
    try {
      const r = await fetch(
        `${apiBase()}/api/eval/threads/${encodeURIComponent(e.thread_id)}/run`,
        { method: 'POST' },
      );
      if (!r.ok) throw new Error(await r.text());
      await onRerun();
    } catch (ex) {
      setRerunErr(ex instanceof Error ? ex.message : String(ex));
    } finally {
      setRerunning(false);
    }
  };

  const handleDelete = () => {
    if (
      window.confirm(
        `Delete eval(s) for thread ${e.thread_id}?\nThis removes every stored eval row for this thread.`,
      )
    ) {
      void onDelete();
    }
  };

  return (
    <div className="flex flex-col gap-4 p-5">
      <header className="flex items-center justify-between gap-3">
        <div>
          <div className="text-xs uppercase tracking-wide text-muted-foreground">
            thread <span className="font-mono">{e.thread_id}</span>
          </div>
          <div className="mt-1 text-sm">
            <span className="text-muted-foreground">topology:</span>{' '}
            <span className="font-mono">{e.topology || 'solo'}</span>
            <span className="mx-2 text-muted-foreground">·</span>
            <span className="text-muted-foreground">judges:</span>{' '}
            <span className="font-mono text-[11px]">
              {e.judge_models?.outcome} / {e.judge_models?.trajectory}
            </span>
            <span className="mx-2 text-muted-foreground">·</span>
            <span className="text-muted-foreground">eval took:</span>{' '}
            <span className="font-mono">{e.elapsed_s.toFixed(1)}s</span>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={rerun}
            disabled={rerunning}
            className="flex items-center gap-2 rounded-md border border-border px-3 py-1.5 text-xs hover:bg-accent disabled:opacity-40"
          >
            {rerunning ? (
              <Loader2 className="h-3 w-3 animate-spin" />
            ) : (
              <RefreshCcw className="h-3 w-3" />
            )}
            re-run rubrics
          </button>
          <button
            type="button"
            onClick={handleDelete}
            className="flex items-center gap-2 rounded-md border border-border px-3 py-1.5 text-xs text-destructive hover:bg-destructive/10"
            title="delete every stored eval for this thread"
          >
            <Trash2 className="h-3 w-3" />
            delete
          </button>
        </div>
      </header>

      {turnCount > 1 && (
        <nav className="flex items-center justify-between gap-2 rounded-md border border-border bg-card/60 px-3 py-2 text-xs">
          <button
            type="button"
            onClick={() => prevTurn && onPickTurn(prevTurn.ts)}
            disabled={!prevTurn}
            className="flex items-center gap-1 rounded-md border border-border px-2 py-1 hover:bg-accent disabled:cursor-not-allowed disabled:opacity-30"
            title={
              prevTurn
                ? `${prevTurn.user_task || '(no task)'}`.slice(0, 80)
                : 'no earlier turn'
            }
          >
            <ChevronLeft className="h-3 w-3" /> prev turn
          </button>
          <div className="flex flex-col items-center text-muted-foreground">
            <span className="font-mono text-foreground">
              turn {turnPosition + 1} of {turnCount}
            </span>
            <span className="text-[10px]">
              click any row in the sidebar to jump
            </span>
          </div>
          <button
            type="button"
            onClick={() => nextTurn && onPickTurn(nextTurn.ts)}
            disabled={!nextTurn}
            className="flex items-center gap-1 rounded-md border border-border px-2 py-1 hover:bg-accent disabled:cursor-not-allowed disabled:opacity-30"
            title={
              nextTurn
                ? `${nextTurn.user_task || '(no task)'}`.slice(0, 80)
                : 'no later turn'
            }
          >
            next turn <ChevronRight className="h-3 w-3" />
          </button>
        </nav>
      )}

      {rerunErr && (
        <div className="rounded-md bg-destructive/10 p-2 text-xs text-destructive">
          {rerunErr}
        </div>
      )}
      {e.error && (
        <div className="rounded-md bg-amber-500/10 p-2 text-xs text-amber-300">
          eval reported errors: {e.error}
        </div>
      )}

      <section className="rounded-md border border-border bg-card p-3">
        <div className="mb-1 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
          user request
        </div>
        <div className="whitespace-pre-wrap text-sm leading-relaxed">
          {e.user_task || '(none)'}
        </div>
      </section>

      <section className="rounded-md border border-border bg-card p-3">
        <div className="mb-1 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
          final answer
        </div>
        <div className="whitespace-pre-wrap text-sm leading-relaxed">
          {e.final_answer || '(empty)'}
        </div>
      </section>

      <div className="grid grid-cols-2 gap-3">
        <RubricCard title="Outcome" score={e.outcome} />
        <RubricCard title="Trajectory" score={e.trajectory} />
      </div>

      <section className="rounded-md border border-border">
        <header className="border-b border-border bg-card/80 px-3 py-1.5 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
          trajectory ({e.tool_calls.length} call{e.tool_calls.length === 1 ? '' : 's'})
        </header>
        <ol className="divide-y divide-border/60">
          {e.tool_calls.length === 0 && (
            <li className="px-3 py-3 text-xs text-muted-foreground">
              no tool calls in this thread.
            </li>
          )}
          {e.tool_calls.map((t, i) => (
            <ToolCallRow key={i} idx={i} call={t} />
          ))}
        </ol>
      </section>
    </div>
  );
}

function RubricCard({
  title,
  score,
}: {
  title: string;
  score: RubricScore;
}): JSX.Element {
  // Pull out the sub-scores (everything that's an int, except `overall`).
  const sub = Object.entries(score).filter(
    ([k, v]) =>
      k !== 'overall' && k !== 'rationale' && typeof v === 'number',
  ) as [string, number][];
  const overall = score.overall;
  return (
    <section className="rounded-md border border-border bg-card p-3">
      <header className="mb-2 flex items-center justify-between">
        <div className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
          {title}
        </div>
        <ScorePill label="overall" value={overall} title={title} />
      </header>
      {sub.length > 0 && (
        <dl className="mb-2 grid grid-cols-3 gap-2 text-xs">
          {sub.map(([k, v]) => (
            <div key={k} className="flex flex-col rounded bg-background/40 p-1.5">
              <dt className="text-[10px] uppercase tracking-wide text-muted-foreground">
                {k}
              </dt>
              <dd className="font-mono text-sm">{v.toFixed(0)} / 5</dd>
            </div>
          ))}
        </dl>
      )}
      {score.rationale && (
        <p className="whitespace-pre-wrap text-xs leading-relaxed text-muted-foreground">
          {score.rationale}
        </p>
      )}
      {!score.overall && !score.rationale && (
        <p className="text-xs text-muted-foreground">
          (no result — check the eval error message above)
        </p>
      )}
    </section>
  );
}

function ToolCallRow({
  idx,
  call,
}: {
  idx: number;
  call: ToolCall;
}): JSX.Element {
  const [open, setOpen] = useState(false);
  return (
    <li className="px-3 py-1.5 text-xs">
      <button
        type="button"
        onClick={() => setOpen((x) => !x)}
        className="flex w-full items-center gap-2 text-left"
      >
        {open ? (
          <ChevronDown className="h-3 w-3 shrink-0 text-muted-foreground" />
        ) : (
          <ChevronRight className="h-3 w-3 shrink-0 text-muted-foreground" />
        )}
        <span className="w-6 shrink-0 text-right font-mono text-muted-foreground">
          {idx}
        </span>
        <Wrench className="h-3 w-3 shrink-0 text-muted-foreground" />
        <span className="font-mono font-medium">{call.tool}</span>
        {call.agent && call.agent !== 'main' && (
          <span className="rounded-sm bg-accent px-1 py-px text-[10px] text-muted-foreground">
            {call.agent}
          </span>
        )}
        <span className="ml-1 flex-1 truncate text-muted-foreground">
          {oneLineArgs(call.args)}
        </span>
        <span
          className={cn(
            'shrink-0 font-mono text-[10px]',
            call.ok === false
              ? 'text-destructive'
              : call.ok === true
                ? 'text-emerald-400'
                : 'text-muted-foreground',
          )}
        >
          {call.ok === false ? 'err' : call.ok === true ? 'ok' : '?'}
        </span>
      </button>
      {open && (
        <div className="ml-9 mt-1 space-y-2 pb-1">
          <pre className="max-h-40 overflow-auto rounded bg-background/60 p-2 font-mono text-[11px]">
            {Object.keys(call.args).length === 0
              ? '(no args)'
              : JSON.stringify(call.args, null, 2)}
          </pre>
          {call.preview && (
            <pre
              className={cn(
                'max-h-40 overflow-auto whitespace-pre-wrap rounded bg-background/60 p-2 font-mono text-[11px]',
                call.ok === false && 'text-destructive',
              )}
            >
              {call.preview}
            </pre>
          )}
        </div>
      )}
    </li>
  );
}

// ---------------------------------------------------------------- helpers ----

function oneLineArgs(args: Record<string, unknown>): string {
  const keys = Object.keys(args);
  if (keys.length === 0) return '';
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
  return line.length > 80 ? line.slice(0, 77) + '…' : line;
}

function relative(iso: string): string {
  // Cheap relative-time stamp. Good enough for "5m ago" / "2h ago" in a list.
  try {
    const then = new Date(iso).getTime();
    const now = Date.now();
    const delta = Math.max(0, (now - then) / 1000);
    if (delta < 60) return `${Math.floor(delta)}s ago`;
    if (delta < 3600) return `${Math.floor(delta / 60)}m ago`;
    if (delta < 86400) return `${Math.floor(delta / 3600)}h ago`;
    return `${Math.floor(delta / 86400)}d ago`;
  } catch {
    return iso;
  }
}
