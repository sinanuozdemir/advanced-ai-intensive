import { useCallback, useEffect, useMemo, useState } from 'react';
import { api } from '@/lib/api';
import { cn } from '@/lib/cn';
import { Plus, Trash2, RefreshCcw, ChevronDown, ChevronRight } from 'lucide-react';

interface AgentRow {
  name: string;
  model?: string;
  description?: string;
  system_prompt?: string;
  tools?: string[];
}

interface ToolRow {
  name: string;
  description: string;
}

interface ServerGroup {
  name: string;
  tools: ToolRow[];
}

const DEFAULT_MODEL = 'openai/gpt-5.4-nano';
const DEFAULT_SYSTEM_PROMPT =
  'You are a specialist worker. Stay focused on the sub-task the main agent handed you. Return a concise result.';
const DEFAULT_DESCRIPTION =
  'A persistent worker the main agent can delegate sub-tasks to.';

interface FormState {
  description: string;
  model: string;
  system_prompt: string;
  tools: string[];
}

function makeDefaultForm(): FormState {
  return {
    description: DEFAULT_DESCRIPTION,
    model: DEFAULT_MODEL,
    system_prompt: DEFAULT_SYSTEM_PROMPT,
    tools: [],
  };
}

export function AgentsView(): JSX.Element {
  const [rows, setRows] = useState<AgentRow[]>([]);
  const [loading, setLoading] = useState(true);
  const [err, setErr] = useState<string | null>(null);
  const [selected, setSelected] = useState<string | null>(null);
  const [form, setForm] = useState<FormState>(makeDefaultForm());
  const [dirty, setDirty] = useState(false);
  const [newNameOpen, setNewNameOpen] = useState(false);
  const [newNameDraft, setNewNameDraft] = useState('');
  // Live tool inventory, grouped by originating MCP server.
  const [servers, setServers] = useState<ServerGroup[]>([]);
  const [serversErr, setServersErr] = useState<string | null>(null);
  // Per-server collapsed state for the tools picker.
  const [collapsed, setCollapsed] = useState<Record<string, boolean>>({});

  // NOTE: deliberately NOT a dependency on ``selected``. Earlier we
  // auto-cleared the selection when the fetched list didn't contain the
  // selected name — but that fires for unsaved drafts too (the new agent
  // hasn't been written to disk yet), instantly reverting "create" to the
  // empty placeholder. Now we leave selection alone here and rely on the
  // explicit delete flow to clear it when an agent really is gone.
  const refresh = useCallback(async () => {
    setLoading(true);
    setErr(null);
    try {
      const agents = (await api.agents()) as unknown as AgentRow[];
      setRows(Array.isArray(agents) ? agents : []);
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }, []);

  const refreshTools = useCallback(async () => {
    setServersErr(null);
    try {
      const r = await api.tools();
      setServers(r.servers ?? []);
    } catch (e) {
      setServersErr(e instanceof Error ? e.message : String(e));
    }
  }, []);

  useEffect(() => {
    void refresh();
    void refreshTools();
  }, [refresh, refreshTools]);

  // Rehydrate the form from the selected row. We deliberately do NOT
  // depend on ``rows`` here — re-running this on every list refresh would
  // blow away in-progress edits (and was the bug that made "create" look
  // like a no-op for brand-new agents whose row isn't on the server yet).
  // ``selected`` changing is itself a user action, so it's OK if that
  // overrides local edits.
  useEffect(() => {
    if (selected === null) {
      setForm(makeDefaultForm());
      setDirty(false);
      return;
    }
    const row = rows.find((r) => r.name === selected);
    if (row) {
      setForm({
        description: row.description ?? '',
        model: row.model ?? DEFAULT_MODEL,
        system_prompt: row.system_prompt ?? '',
        tools: row.tools ?? [],
      });
      setDirty(false);
    }
    // Intentional: rows omitted from deps to avoid clobbering edits.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selected]);

  // Set of valid tool names (the live ones the engine actually loaded).
  // We use this to flag stale tool references on an existing agent's spec.
  const validToolNames = useMemo(() => {
    const out = new Set<string>();
    for (const s of servers) for (const t of s.tools) out.add(t.name);
    return out;
  }, [servers]);

  const staleTools = useMemo(
    () => form.tools.filter((t) => !validToolNames.has(t)),
    [form.tools, validToolNames],
  );

  const startNew = () => {
    setNewNameDraft('');
    setNewNameOpen(true);
  };

  const confirmNewName = () => {
    const name = newNameDraft.trim();
    if (!name) return;
    if (!/^[a-z0-9][a-z0-9_-]*$/i.test(name)) {
      window.alert(
        'agent name must be a slug — letters, digits, underscore, or dash; no spaces.',
      );
      return;
    }
    if (rows.some((r) => r.name === name)) {
      window.alert(`agent ${name} already exists`);
      return;
    }
    setNewNameOpen(false);
    setSelected(name);
    setForm(makeDefaultForm());
    setDirty(true);
  };

  const save = async () => {
    if (!selected) return;
    if (!form.description.trim()) {
      window.alert('description is required.');
      return;
    }
    if (!form.system_prompt.trim()) {
      window.alert('system prompt is required.');
      return;
    }
    try {
      await api.putAgent(selected, {
        description: form.description.trim(),
        model: form.model.trim() || DEFAULT_MODEL,
        system_prompt: form.system_prompt.trim(),
        tools: form.tools,
      });
      setDirty(false);
      await refresh();
    } catch (e) {
      window.alert(`save failed: ${e instanceof Error ? e.message : e}`);
    }
  };

  const remove = async (name: string) => {
    if (!window.confirm(`delete agent ${name}?`)) return;
    try {
      await api.deleteAgent(name);
      if (selected === name) setSelected(null);
      await refresh();
    } catch (e) {
      window.alert(`delete failed: ${e instanceof Error ? e.message : e}`);
    }
  };

  const updateForm = <K extends keyof FormState>(key: K, value: FormState[K]) => {
    setForm((prev) => ({ ...prev, [key]: value }));
    setDirty(true);
  };

  const toggleTool = (name: string) => {
    setForm((prev) => {
      const has = prev.tools.includes(name);
      return {
        ...prev,
        tools: has ? prev.tools.filter((t) => t !== name) : [...prev.tools, name],
      };
    });
    setDirty(true);
  };

  const toggleServer = (group: ServerGroup, allSelected: boolean) => {
    setForm((prev) => {
      const groupNames = new Set(group.tools.map((t) => t.name));
      const remaining = prev.tools.filter((t) => !groupNames.has(t));
      return {
        ...prev,
        tools: allSelected ? remaining : [...remaining, ...group.tools.map((t) => t.name)],
      };
    });
    setDirty(true);
  };

  return (
    <div className="flex h-full">
      {/* List */}
      <aside className="flex w-72 flex-col border-r border-border">
        <div className="flex items-center justify-between border-b border-border px-3 py-2 text-sm">
          <span className="font-semibold">persistent agents</span>
          <div className="flex gap-1">
            <button
              type="button"
              onClick={() => {
                void refresh();
                void refreshTools();
              }}
              className="rounded p-1 text-muted-foreground hover:bg-accent hover:text-foreground"
              title="refresh"
            >
              <RefreshCcw className="h-3.5 w-3.5" />
            </button>
            <button
              type="button"
              onClick={startNew}
              className="rounded p-1 text-muted-foreground hover:bg-accent hover:text-foreground"
              title="new agent"
            >
              <Plus className="h-3.5 w-3.5" />
            </button>
          </div>
        </div>
        <ul className="flex-1 overflow-y-auto">
          {loading && (
            <li className="p-3 text-xs text-muted-foreground">loading…</li>
          )}
          {err && (
            <li className="p-3 text-xs text-destructive">{err}</li>
          )}
          {/* Unsaved draft sentinel — shown when the user has clicked "create"
              but hasn't saved yet, so the agent has no .toml on disk. */}
          {!loading && selected && !rows.some((r) => r.name === selected) && (
            <li>
              <button
                type="button"
                onClick={() => setSelected(selected)}
                className="flex w-full items-center justify-between bg-accent px-3 py-2 text-left text-sm"
              >
                <div className="min-w-0">
                  <div className="truncate font-medium">{selected}</div>
                  <div className="truncate text-xs italic text-muted-foreground">
                    unsaved draft
                  </div>
                </div>
                <Trash2
                  className="h-3.5 w-3.5 text-muted-foreground hover:text-destructive"
                  onClick={(e) => {
                    e.stopPropagation();
                    if (
                      !dirty ||
                      window.confirm('discard unsaved draft?')
                    ) {
                      setSelected(null);
                    }
                  }}
                />
              </button>
            </li>
          )}
          {!loading &&
            rows.map((row) => (
              <li key={row.name}>
                <button
                  type="button"
                  onClick={() => setSelected(row.name)}
                  className={cn(
                    'flex w-full items-center justify-between px-3 py-2 text-left text-sm hover:bg-accent',
                    selected === row.name && 'bg-accent',
                  )}
                >
                  <div className="min-w-0">
                    <div className="truncate font-medium">{row.name}</div>
                    {row.description && (
                      <div className="truncate text-xs text-muted-foreground">
                        {row.description}
                      </div>
                    )}
                  </div>
                  <Trash2
                    className="h-3.5 w-3.5 text-muted-foreground hover:text-destructive"
                    onClick={(e) => {
                      e.stopPropagation();
                      void remove(row.name);
                    }}
                  />
                </button>
              </li>
            ))}
          {!loading && rows.length === 0 && !err && (
            <li className="p-3 text-xs text-muted-foreground">
              no persistent agents yet
            </li>
          )}
        </ul>
      </aside>

      {/* Editor */}
      <section className="flex flex-1 flex-col">
        {selected === null ? (
          <div className="flex h-full flex-col items-center justify-center gap-3 px-6 text-center text-sm text-muted-foreground">
            <p>
              select an agent, or{' '}
              <button className="underline" onClick={startNew}>
                create a new one
              </button>
            </p>
            <p className="max-w-md text-xs">
              Persistent agents are TOML specs in{' '}
              <code className="rounded bg-accent px-1 py-px">
                .forge/agents/
              </code>{' '}
              that the main agent can call as <code>delegate_to_&lt;name&gt;</code>{' '}
              tools. Give one a focused system prompt and a deny-by-default
              tool allowlist; the main agent will see its description in
              the specialist menu and call it when a sub-task fits.
            </p>
          </div>
        ) : (
          <>
            <div className="flex items-center justify-between border-b border-border px-4 py-2">
              <div className="text-sm font-semibold">{selected}</div>
              <button
                type="button"
                onClick={save}
                disabled={!dirty}
                className={cn(
                  'rounded-md bg-primary px-3 py-1 text-xs text-primary-foreground',
                  !dirty && 'opacity-40',
                )}
              >
                save
              </button>
            </div>
            <div className="flex flex-1 flex-col gap-4 overflow-y-auto p-4">
              <label className="flex flex-col gap-1 text-xs">
                <span className="font-medium text-muted-foreground">
                  description
                </span>
                <input
                  className="rounded-md border border-input bg-background px-2 py-1.5 text-sm outline-none focus:ring-2 focus:ring-ring"
                  value={form.description}
                  onChange={(e) => updateForm('description', e.target.value)}
                  placeholder="What does this worker do? The main agent reads this to decide whether to call delegate_to_<name>."
                />
              </label>

              <label className="flex flex-col gap-1 text-xs">
                <span className="font-medium text-muted-foreground">model</span>
                <input
                  className="rounded-md border border-input bg-background px-2 py-1.5 font-mono text-xs outline-none focus:ring-2 focus:ring-ring"
                  value={form.model}
                  onChange={(e) => updateForm('model', e.target.value)}
                  placeholder={DEFAULT_MODEL}
                />
              </label>

              <label className="flex flex-col gap-1 text-xs">
                <span className="font-medium text-muted-foreground">
                  system prompt
                </span>
                <textarea
                  className="min-h-[140px] resize-y rounded-md border border-input bg-background p-2 font-mono text-xs outline-none focus:ring-2 focus:ring-ring"
                  spellCheck={false}
                  value={form.system_prompt}
                  onChange={(e) => updateForm('system_prompt', e.target.value)}
                />
              </label>

              <div className="flex flex-col gap-2">
                <div className="flex items-center justify-between">
                  <span className="text-xs font-medium text-muted-foreground">
                    tools{' '}
                    <span className="text-muted-foreground/70">
                      ({form.tools.length} selected)
                    </span>
                  </span>
                  <button
                    type="button"
                    onClick={() => void refreshTools()}
                    className="text-xs text-muted-foreground hover:text-foreground"
                  >
                    refresh inventory
                  </button>
                </div>
                {serversErr && (
                  <div className="rounded-md border border-destructive/50 bg-destructive/10 px-2 py-1 text-xs text-destructive">
                    failed to load tool inventory: {serversErr}
                  </div>
                )}
                {staleTools.length > 0 && (
                  <div className="rounded-md border border-amber-500/50 bg-amber-500/10 px-2 py-1 text-xs">
                    <span className="font-medium">stale tool refs:</span>{' '}
                    {staleTools.join(', ')}{' '}
                    <span className="text-muted-foreground">
                      (not loaded by the engine; will be saved verbatim but
                      won't gate anything until the matching MCP server is
                      installed)
                    </span>
                  </div>
                )}
                <div className="rounded-md border border-border">
                  {servers.length === 0 && !serversErr && (
                    <div className="p-3 text-xs text-muted-foreground">
                      no tools loaded yet — start a chat to boot the engine,
                      then hit "refresh inventory".
                    </div>
                  )}
                  {servers.map((group) => {
                    const groupNames = group.tools.map((t) => t.name);
                    const selectedCount = groupNames.filter((n) =>
                      form.tools.includes(n),
                    ).length;
                    const allSelected =
                      selectedCount === groupNames.length && groupNames.length > 0;
                    const isCollapsed = collapsed[group.name] ?? true;
                    return (
                      <div
                        key={group.name}
                        className="border-b border-border last:border-b-0"
                      >
                        <div className="flex items-center justify-between px-2 py-1.5">
                          <button
                            type="button"
                            onClick={() =>
                              setCollapsed((prev) => ({
                                ...prev,
                                [group.name]: !isCollapsed,
                              }))
                            }
                            className="flex flex-1 items-center gap-1 text-left text-xs"
                          >
                            {isCollapsed ? (
                              <ChevronRight className="h-3 w-3" />
                            ) : (
                              <ChevronDown className="h-3 w-3" />
                            )}
                            <span className="font-mono font-medium">
                              {group.name}
                            </span>
                            <span className="text-muted-foreground">
                              ({selectedCount}/{groupNames.length})
                            </span>
                          </button>
                          <button
                            type="button"
                            onClick={() => toggleServer(group, allSelected)}
                            className="rounded px-2 py-0.5 text-xs text-muted-foreground hover:bg-accent hover:text-foreground"
                          >
                            {allSelected ? 'none' : 'all'}
                          </button>
                        </div>
                        {!isCollapsed && (
                          <ul className="px-2 pb-2">
                            {group.tools.map((tool) => {
                              const checked = form.tools.includes(tool.name);
                              return (
                                <li key={tool.name}>
                                  <label className="flex items-start gap-2 rounded px-2 py-1 hover:bg-accent">
                                    <input
                                      type="checkbox"
                                      checked={checked}
                                      onChange={() => toggleTool(tool.name)}
                                      className="mt-0.5"
                                    />
                                    <div className="min-w-0">
                                      <div className="font-mono text-xs">
                                        {tool.name}
                                      </div>
                                      {tool.description && (
                                        <div className="text-xs text-muted-foreground">
                                          {tool.description}
                                        </div>
                                      )}
                                    </div>
                                  </label>
                                </li>
                              );
                            })}
                          </ul>
                        )}
                      </div>
                    );
                  })}
                </div>
              </div>
            </div>
          </>
        )}
      </section>

      {newNameOpen && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-4"
          onClick={() => setNewNameOpen(false)}
        >
          <div
            className="w-full max-w-sm rounded-lg border border-border bg-card p-4 shadow-xl"
            onClick={(e) => e.stopPropagation()}
          >
            <h2 className="mb-2 text-sm font-semibold">
              new persistent agent
            </h2>
            <p className="mb-3 text-xs text-muted-foreground">
              Pick a slug (letters, digits, dash, underscore — no spaces).
              The file will be saved at{' '}
              <code className="rounded bg-accent px-1 py-px">
                .forge/agents/&lt;slug&gt;.toml
              </code>
              .
            </p>
            <input
              autoFocus
              value={newNameDraft}
              onChange={(e) => setNewNameDraft(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter') confirmNewName();
                if (e.key === 'Escape') setNewNameOpen(false);
              }}
              placeholder="e.g. researcher"
              className="w-full rounded-md border border-input bg-background px-2 py-1.5 text-sm outline-none focus:ring-2 focus:ring-ring"
            />
            <div className="mt-3 flex justify-end gap-2">
              <button
                type="button"
                onClick={() => setNewNameOpen(false)}
                className="rounded-md px-3 py-1 text-xs text-muted-foreground hover:bg-accent"
              >
                cancel
              </button>
              <button
                type="button"
                onClick={confirmNewName}
                disabled={!newNameDraft.trim()}
                className="rounded-md bg-primary px-3 py-1 text-xs text-primary-foreground disabled:opacity-40"
              >
                create
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
