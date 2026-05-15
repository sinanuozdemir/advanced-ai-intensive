import { useCallback, useEffect, useState } from 'react';
import { api } from '@/lib/api';
import { cn } from '@/lib/cn';
import { Search, Trash2, Loader2 } from 'lucide-react';

type Tab = 'semantic' | 'episodic' | 'procedural';

type SemanticHit = {
  id: string;
  text: string;
  score: number;
  thread_id?: string;
  created_at?: string;
};
type EpisodicRow = {
  summary: string;
  thread_id: string;
  score: number;
  created_at: string;
  updated_at?: string;
  id: string;
};
type ProceduralRow = {
  name: string;
  fragment: string;
  when_to_use: string;
  usage_count: number;
  score: number;
  created_at: string;
};

function formatTimestamp(iso: string): string {
  try {
    return new Date(iso).toLocaleString();
  } catch {
    return iso;
  }
}

export function MemoryView(): JSX.Element {
  const [tab, setTab] = useState<Tab>('semantic');
  return (
    <div className="flex h-full flex-col">
      <div className="flex gap-1 border-b border-border bg-card px-2 py-1">
        {(['semantic', 'episodic', 'procedural'] as Tab[]).map((t) => (
          <button
            key={t}
            type="button"
            onClick={() => setTab(t)}
            className={cn(
              'rounded-md px-3 py-1 text-xs uppercase tracking-wide',
              t === tab
                ? 'bg-primary/15 text-primary'
                : 'text-muted-foreground hover:bg-accent hover:text-foreground',
            )}
          >
            {t}
          </button>
        ))}
      </div>
      <div className="flex-1 overflow-y-auto">
        {tab === 'semantic' && <SemanticPanel />}
        {tab === 'episodic' && <EpisodicPanel />}
        {tab === 'procedural' && <ProceduralPanel />}
      </div>
    </div>
  );
}

// ----------------------------------------------------------------- semantic -

function SemanticPanel(): JSX.Element {
  const [q, setQ] = useState('');
  const [hits, setHits] = useState<SemanticHit[]>([]);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  // When ``mode === 'browse'`` we show the full list newest-first; when
  // ``'search'`` we show similarity-ranked results for the last query.
  const [mode, setMode] = useState<'browse' | 'search'>('browse');
  // ``deletingId`` tracks the row currently being removed so we can
  // show a spinner on its trash button and disable double-clicks.
  const [deletingId, setDeletingId] = useState<string | null>(null);

  const browse = useCallback(async () => {
    setLoading(true);
    setErr(null);
    try {
      const r = await api.semanticBrowse(10);
      setHits(Array.isArray(r) ? r : []);
      setMode('browse');
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }, []);

  const search = useCallback(async () => {
    if (!q.trim()) {
      // Empty query = back to browse mode.
      void browse();
      return;
    }
    setLoading(true);
    setErr(null);
    try {
      const r = await api.semanticSearch(q.trim(), 15);
      setHits(Array.isArray(r) ? r : []);
      setMode('search');
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }, [q, browse]);

  useEffect(() => {
    void browse();
  }, [browse]);

  const onDelete = useCallback(
    async (id: string, preview: string) => {
      if (!id) return;
      const confirmed = window.confirm(
        `delete this semantic memory?\n\n"${preview.slice(0, 120)}${
          preview.length > 120 ? '…' : ''
        }"\n\nthis removes it from the vector store and cannot be undone.`,
      );
      if (!confirmed) return;
      setDeletingId(id);
      setErr(null);
      try {
        await api.deleteSemantic(id);
        // Optimistic local drop — avoids the flash of a full re-fetch.
        setHits((prev) => prev.filter((h) => h.id !== id));
      } catch (e) {
        setErr(e instanceof Error ? e.message : String(e));
      } finally {
        setDeletingId(null);
      }
    },
    [],
  );

  const sectionLabel =
    mode === 'search'
      ? `${hits.length} result${hits.length === 1 ? '' : 's'} for "${q}"`
      : `${hits.length} most recent ${hits.length === 1 ? 'memory' : 'memories'}`;

  return (
    <div className="p-4">
      <div className="mx-auto flex max-w-2xl items-center gap-2">
        <div className="relative flex-1">
          <Search className="absolute left-2 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
          <input
            value={q}
            onChange={(e) => setQ(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && search()}
            placeholder="semantic search… (empty + Enter = newest 10)"
            className="w-full rounded-md border border-input bg-background py-2 pl-8 pr-3 text-sm outline-none focus:ring-2 focus:ring-ring"
          />
        </div>
        <button
          type="button"
          onClick={search}
          disabled={loading}
          className="rounded-md bg-primary px-3 py-2 text-sm text-primary-foreground disabled:opacity-40"
        >
          search
        </button>
      </div>
      <div className="mx-auto mt-4 max-w-2xl space-y-2">
        {err && <div className="text-sm text-destructive">{err}</div>}
        {!err && (
          <div className="px-1 text-[10px] uppercase tracking-wide text-muted-foreground">
            {sectionLabel}
          </div>
        )}
        {hits.map((h, i) => (
          <article
            key={`${h.id}-${i}`}
            className="group rounded-md border border-border bg-card p-3 text-sm"
          >
            <div className="mb-1 flex items-center justify-between gap-2 text-xs text-muted-foreground">
              <span className="font-mono">{h.id}</span>
              <div className="flex items-center gap-2">
                <span>
                  {mode === 'search'
                    ? `score ${h.score.toFixed(3)}`
                    : h.created_at
                      ? formatTimestamp(h.created_at)
                      : ''}
                </span>
                <button
                  type="button"
                  onClick={() => onDelete(h.id, h.text)}
                  disabled={deletingId === h.id}
                  title="delete this memory"
                  className={cn(
                    'rounded-md p-1 transition-colors',
                    'text-muted-foreground hover:bg-destructive/10 hover:text-destructive',
                    'opacity-0 group-hover:opacity-100',
                    deletingId === h.id && 'opacity-100',
                  )}
                >
                  {deletingId === h.id ? (
                    <Loader2 className="h-3.5 w-3.5 animate-spin" />
                  ) : (
                    <Trash2 className="h-3.5 w-3.5" />
                  )}
                </button>
              </div>
            </div>
            <p className="whitespace-pre-wrap leading-relaxed">{h.text}</p>
          </article>
        ))}
        {!loading && hits.length === 0 && !err && (
          <p className="text-center text-xs text-muted-foreground">
            no memories yet — the agent will write durable facts here when it
            calls `semantic_write`
          </p>
        )}
      </div>
    </div>
  );
}

// ------------------------------------------------------------------ episodic

function EpisodicPanel(): JSX.Element {
  const [episodes, setEpisodes] = useState<EpisodicRow[]>([]);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    let alive = true;
    api
      .episodic(50)
      .then((r) => alive && setEpisodes(Array.isArray(r) ? r : []))
      .catch((e) => alive && setErr(e instanceof Error ? e.message : String(e)));
    return () => {
      alive = false;
    };
  }, []);

  return (
    <div className="p-4">
      {err && <div className="text-sm text-destructive">{err}</div>}
      <div className="mx-auto max-w-3xl space-y-2">
        {episodes.map((ep) => (
          <article
            key={ep.id}
            className="rounded-md border border-border bg-card p-3 text-sm"
          >
            <div className="mb-1 flex justify-between gap-2 text-xs text-muted-foreground">
              <span className="font-mono">{ep.thread_id}</span>
              <span title={`created ${ep.created_at}`}>
                updated {formatTimestamp(ep.updated_at || ep.created_at)}
              </span>
            </div>
            <p className="whitespace-pre-wrap leading-relaxed">{ep.summary}</p>
          </article>
        ))}
        {episodes.length === 0 && !err && (
          <p className="text-center text-xs text-muted-foreground">
            no episodes yet — reflection upserts one rolling summary per thread after each turn
          </p>
        )}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------- procedural

function ProceduralPanel(): JSX.Element {
  const [skills, setSkills] = useState<ProceduralRow[]>([]);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    let alive = true;
    api
      .procedural()
      .then((r) => alive && setSkills(Array.isArray(r) ? r : []))
      .catch((e) => alive && setErr(e instanceof Error ? e.message : String(e)));
    return () => {
      alive = false;
    };
  }, []);

  return (
    <div className="p-4">
      {err && <div className="text-sm text-destructive">{err}</div>}
      <div className="mx-auto max-w-3xl space-y-2">
        {skills.map((sk) => (
          <article
            key={`${sk.name}-${sk.created_at}`}
            className="rounded-md border border-border bg-card p-3 text-sm"
          >
            <div className="mb-1 flex items-baseline justify-between gap-2">
              <span className="font-mono text-xs text-primary">{sk.name}</span>
              <span className="text-xs text-muted-foreground">
                {formatTimestamp(sk.created_at)}
              </span>
            </div>
            <p className="whitespace-pre-wrap leading-relaxed">{sk.fragment}</p>
            {sk.when_to_use && (
              <p className="mt-1 text-xs italic text-muted-foreground">
                when: {sk.when_to_use}
              </p>
            )}
          </article>
        ))}
        {skills.length === 0 && !err && (
          <p className="text-center text-xs text-muted-foreground">
            no distilled skills yet
          </p>
        )}
      </div>
    </div>
  );
}
