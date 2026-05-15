import { useCallback, useEffect, useRef, useState } from 'react';
import { apiBase } from '@/lib/api';
import { cn } from '@/lib/cn';
import {
  UploadCloud,
  CheckCircle2,
  AlertCircle,
  Loader2,
  RefreshCcw,
  Trash2,
} from 'lucide-react';

// MCP management view. Three panels (drag-drop, validation, server list).
//
// This depends on three backend endpoints that are part of the MCP todos:
//   - POST /api/mcp/validate
//   - POST /api/mcp/install
//   - GET  /api/mcp
// Until those land the validate/install fetches will fail with 404; the UI
// degrades gracefully (the user just sees an error banner).

interface McpServer {
  name: string;
  kind: 'builtin' | 'user_json';
  source?: string;
  description?: string;
  tools?: string[];
  pending_restart?: boolean;
}

interface ValidateResult {
  ok: boolean;
  tools?: Array<{ name: string; description?: string }>;
  error?: string;
}

export function McpView(): JSX.Element {
  const [servers, setServers] = useState<McpServer[]>([]);
  const [listErr, setListErr] = useState<string | null>(null);
  const [pendingRestart, setPendingRestart] = useState(false);
  const [reloading, setReloading] = useState(false);
  const [reloadMsg, setReloadMsg] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    try {
      const r = await fetch(`${apiBase()}/api/mcp`);
      if (!r.ok) throw new Error(`${r.status}`);
      const data = (await r.json()) as { servers: McpServer[] };
      setServers(data.servers ?? []);
      setPendingRestart((data.servers ?? []).some((s) => s.pending_restart));
      setListErr(null);
    } catch (e) {
      setListErr(e instanceof Error ? e.message : String(e));
    }
  }, []);

  const reload = useCallback(async () => {
    setReloading(true);
    setReloadMsg(null);
    try {
      const r = await fetch(`${apiBase()}/api/mcp/reload`, { method: 'POST' });
      if (!r.ok) throw new Error(await r.text());
      const data = (await r.json()) as { tool_count_after: number };
      setReloadMsg(`engine rebooted · ${data.tool_count_after} tools live`);
      await refresh();
    } catch (e) {
      setReloadMsg(`reload failed: ${e instanceof Error ? e.message : String(e)}`);
    } finally {
      setReloading(false);
    }
  }, [refresh]);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  return (
    <div className="flex h-full flex-col">
      {pendingRestart && (
        <div className="flex items-center justify-between gap-3 border-b border-amber-500/40 bg-amber-500/10 px-4 py-2 text-sm text-amber-300">
          <span>
            MCP servers installed — reload the engine for the new tools to
            become callable.
          </span>
          <button
            type="button"
            onClick={reload}
            disabled={reloading}
            className="rounded-md border border-amber-500/60 px-3 py-1 text-xs font-semibold text-amber-200 hover:bg-amber-500/20 disabled:opacity-40"
          >
            {reloading ? 'reloading…' : 'reload engine'}
          </button>
        </div>
      )}
      {reloadMsg && (
        <div className="border-b border-border bg-card px-4 py-1.5 text-xs text-muted-foreground">
          {reloadMsg}
        </div>
      )}
      <div className="grid flex-1 grid-cols-2 overflow-hidden">
        <DropZone onInstalled={refresh} />
        <ServerList servers={servers} listErr={listErr} onRefresh={refresh} />
      </div>
    </div>
  );
}

// ----------------------------------------------------------------- drop zone

function DropZone({ onInstalled }: { onInstalled: () => void }): JSX.Element {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragOver, setDragOver] = useState(false);
  const [staged, setStaged] = useState<File | null>(null);
  const [validation, setValidation] = useState<ValidateResult | null>(null);
  const [validating, setValidating] = useState(false);
  const [installName, setInstallName] = useState('');
  const [installing, setInstalling] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const onFile = (file: File) => {
    setStaged(file);
    setValidation(null);
    setInstallName(file.name.replace(/\.json$/i, ''));
    setError(null);
  };

  const validate = async () => {
    if (!staged) return;
    setValidating(true);
    setError(null);
    try {
      const body = await staged.text();
      const r = await fetch(`${apiBase()}/api/mcp/validate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          filename: staged.name,
          contents: body,
        }),
      });
      if (!r.ok) throw new Error(await r.text());
      setValidation((await r.json()) as ValidateResult);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setValidating(false);
    }
  };

  const install = async () => {
    if (!staged || !installName.trim() || !validation?.ok) return;
    setInstalling(true);
    setError(null);
    try {
      const body = await staged.text();
      const r = await fetch(`${apiBase()}/api/mcp/install`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          name: installName.trim(),
          filename: staged.name,
          contents: body,
        }),
      });
      if (!r.ok) throw new Error(await r.text());
      setStaged(null);
      setValidation(null);
      setInstallName('');
      onInstalled();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setInstalling(false);
    }
  };

  return (
    <section
      onDragOver={(e) => {
        e.preventDefault();
        setDragOver(true);
      }}
      onDragLeave={() => setDragOver(false)}
      onDrop={(e) => {
        e.preventDefault();
        setDragOver(false);
        const f = e.dataTransfer.files?.[0];
        if (f) onFile(f);
      }}
      className="flex flex-col border-r border-border"
    >
      <header className="border-b border-border px-4 py-2 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
        add MCP server
      </header>
      <div className="flex flex-1 flex-col gap-3 p-4">
        <div
          className={cn(
            'flex flex-1 cursor-pointer flex-col items-center justify-center rounded-lg border-2 border-dashed text-sm text-muted-foreground transition-colors',
            dragOver ? 'border-primary bg-primary/5' : 'border-border',
          )}
          onClick={() => inputRef.current?.click()}
        >
          <UploadCloud className="mb-2 h-8 w-8" />
          <p>drop a .json MCP descriptor here</p>
          <p className="mt-1 text-xs">
            {'{ "command": …, "args": […], "env": { … } }'}
          </p>
          <p className="mt-1 text-xs">or click to choose</p>
          <input
            ref={inputRef}
            type="file"
            accept=".json,application/json"
            className="hidden"
            onChange={(e) => {
              const f = e.target.files?.[0];
              if (f) onFile(f);
            }}
          />
        </div>

        {staged && (
          <div className="rounded-md border border-border bg-card p-3 text-sm">
            <div className="mb-2 flex items-center justify-between">
              <div>
                <div className="font-medium">{staged.name}</div>
                <div className="text-xs text-muted-foreground">
                  {staged.size.toLocaleString()} bytes · json descriptor
                </div>
              </div>
              <button
                type="button"
                onClick={validate}
                disabled={validating}
                className="rounded-md border border-border px-3 py-1 text-xs hover:bg-accent"
              >
                {validating ? (
                  <Loader2 className="h-3 w-3 animate-spin" />
                ) : (
                  'validate'
                )}
              </button>
            </div>

            {validation && (
              <div
                className={cn(
                  'mt-2 rounded-md p-2 text-xs',
                  validation.ok
                    ? 'bg-emerald-500/10 text-emerald-300'
                    : 'bg-destructive/10 text-destructive',
                )}
              >
                {validation.ok ? (
                  <>
                    <div className="mb-1 flex items-center gap-1">
                      <CheckCircle2 className="h-3 w-3" /> validated ·{' '}
                      {validation.tools?.length ?? 0} tools
                    </div>
                    <ul className="list-disc space-y-0.5 pl-4">
                      {validation.tools?.slice(0, 8).map((t) => (
                        <li key={t.name}>
                          <span className="font-mono">{t.name}</span>
                          {t.description && (
                            <span className="ml-1 opacity-70">
                              — {t.description.slice(0, 80)}
                            </span>
                          )}
                        </li>
                      ))}
                    </ul>
                  </>
                ) : (
                  <div className="flex items-center gap-1">
                    <AlertCircle className="h-3 w-3" />
                    {validation.error}
                  </div>
                )}
              </div>
            )}

            {validation?.ok && (
              <div className="mt-3 flex items-end gap-2">
                <label className="flex-1">
                  <span className="text-xs text-muted-foreground">
                    name (slug)
                  </span>
                  <input
                    value={installName}
                    onChange={(e) => setInstallName(e.target.value)}
                    className="mt-1 w-full rounded-md border border-input bg-background px-2 py-1 text-sm outline-none focus:ring-2 focus:ring-ring"
                  />
                </label>
                <button
                  type="button"
                  onClick={install}
                  disabled={installing || !installName.trim()}
                  className="rounded-md bg-primary px-3 py-1 text-sm text-primary-foreground disabled:opacity-40"
                >
                  {installing ? 'installing…' : 'install'}
                </button>
              </div>
            )}
          </div>
        )}

        {error && (
          <div className="rounded-md bg-destructive/10 p-2 text-xs text-destructive">
            {error}
          </div>
        )}
      </div>
    </section>
  );
}

// -------------------------------------------------------------- server list

function ServerList({
  servers,
  listErr,
  onRefresh,
}: {
  servers: McpServer[];
  listErr: string | null;
  onRefresh: () => void;
}): JSX.Element {
  const [deleting, setDeleting] = useState<string | null>(null);
  const [deleteErr, setDeleteErr] = useState<string | null>(null);

  const uninstall = async (name: string) => {
    if (
      !window.confirm(
        `Uninstall MCP server "${name}"?\n\nThe descriptor will be deleted from .forge/mcp_servers/. The running engine still has its tools loaded — hit "reload engine" afterwards to actually drop them.`,
      )
    ) {
      return;
    }
    setDeleting(name);
    setDeleteErr(null);
    try {
      const r = await fetch(
        `${apiBase()}/api/mcp/${encodeURIComponent(name)}`,
        { method: 'DELETE' },
      );
      if (!r.ok) {
        let detail = `${r.status}`;
        try {
          const body = (await r.json()) as { detail?: string };
          if (body?.detail) detail = body.detail;
        } catch {
          /* ignore */
        }
        throw new Error(detail);
      }
      onRefresh();
    } catch (e) {
      setDeleteErr(e instanceof Error ? e.message : String(e));
    } finally {
      setDeleting(null);
    }
  };

  return (
    <section className="flex flex-col">
      <header className="flex items-center justify-between border-b border-border px-4 py-2 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
        <span>installed servers</span>
        <button
          type="button"
          onClick={onRefresh}
          className="rounded p-1 text-muted-foreground hover:bg-accent hover:text-foreground"
        >
          <RefreshCcw className="h-3 w-3" />
        </button>
      </header>
      <div className="flex-1 overflow-y-auto p-3">
        {listErr && (
          <div className="rounded-md bg-destructive/10 p-2 text-xs text-destructive">
            {listErr}
          </div>
        )}
        {deleteErr && (
          <div className="mb-2 rounded-md bg-destructive/10 p-2 text-xs text-destructive">
            uninstall failed: {deleteErr}
          </div>
        )}
        <ul className="space-y-2">
          {servers.map((s) => (
            <li
              key={s.name}
              className="rounded-md border border-border bg-card p-3 text-sm"
            >
              <div className="flex items-center justify-between gap-2">
                <div className="min-w-0 font-mono font-semibold">{s.name}</div>
                <div className="flex items-center gap-2">
                  <span
                    className={cn(
                      'rounded-sm px-1.5 py-0.5 text-[10px] uppercase',
                      s.kind === 'builtin'
                        ? 'bg-secondary text-muted-foreground'
                        : 'bg-primary/15 text-primary',
                    )}
                  >
                    {s.kind.replace('user_', '')}
                  </span>
                  {s.kind !== 'builtin' && (
                    <button
                      type="button"
                      onClick={() => uninstall(s.name)}
                      disabled={deleting === s.name}
                      title="uninstall"
                      className="rounded p-1 text-muted-foreground hover:bg-destructive/10 hover:text-destructive disabled:opacity-40"
                    >
                      {deleting === s.name ? (
                        <Loader2 className="h-3.5 w-3.5 animate-spin" />
                      ) : (
                        <Trash2 className="h-3.5 w-3.5" />
                      )}
                    </button>
                  )}
                </div>
              </div>
              {s.description && (
                <p className="mt-1 text-xs text-muted-foreground">
                  {s.description}
                </p>
              )}
              {s.tools && s.tools.length > 0 && (
                <p className="mt-1 truncate text-xs font-mono text-muted-foreground">
                  {s.tools.slice(0, 6).join(', ')}
                  {s.tools.length > 6 && ` +${s.tools.length - 6}`}
                </p>
              )}
            </li>
          ))}
        </ul>
      </div>
    </section>
  );
}
