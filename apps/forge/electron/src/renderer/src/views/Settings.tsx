import { useCallback, useEffect, useMemo, useState } from 'react';
import { api, type ModelCatalog } from '@/lib/api';
import { cn } from '@/lib/cn';
import { Save, RefreshCcw, CheckCircle2, AlertCircle, Loader2 } from 'lucide-react';

// Schema-driven config form.
//
// The Pydantic config has well over a dozen fields; rather than hand-coding a
// React form for every one, we render generic inputs from the JSON Schema
// served at `/api/config/schema`. Each leaf field gets an input typed to its
// schema type. Fields marked `live: true` are hot-reloadable; others apply
// only on the next thread.
//
// For *model* fields (named `models.*`), the input grows a "verify" button
// that calls `/api/models/health` so the user can confirm a slug actually
// resolves before saving.

type Schema = Record<string, unknown>;

interface SchemaProp {
  type?: string;
  title?: string;
  description?: string;
  default?: unknown;
  enum?: unknown[];
  $ref?: string;
  // Extracted from json_schema_extra={"live": ...} on the Field.
  live?: boolean;
  items?: SchemaProp;
}

export function SettingsView(): JSX.Element {
  const [schema, setSchema] = useState<Schema | null>(null);
  const [config, setConfig] = useState<Record<string, Record<string, unknown>> | null>(null);
  const [catalog, setCatalog] = useState<ModelCatalog | null>(null);
  const [refreshingCatalog, setRefreshingCatalog] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);
  const [dirty, setDirty] = useState(false);

  const refresh = useCallback(async () => {
    setErr(null);
    try {
      const [s, c] = await Promise.all([api.configSchema(), api.config()]);
      setSchema(s as Schema);
      setConfig(c as Record<string, Record<string, unknown>>);
      setDirty(false);
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    }
  }, []);

  const refreshCatalog = useCallback(async () => {
    setRefreshingCatalog(true);
    try {
      setCatalog(await api.modelCatalog());
    } catch {
      // Non-fatal: the picker falls back to the free-text input.
      setCatalog(null);
    } finally {
      setRefreshingCatalog(false);
    }
  }, []);

  useEffect(() => {
    void refresh();
    void refreshCatalog();
  }, [refresh, refreshCatalog]);

  const save = useCallback(async () => {
    if (!config) return;
    setSaving(true);
    setErr(null);
    try {
      await api.putConfig(config);
      setDirty(false);
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setSaving(false);
    }
  }, [config]);

  const onChange = (
    section: string,
    field: string,
    value: unknown,
  ) => {
    setConfig((cur) => {
      if (!cur) return cur;
      const next = { ...cur, [section]: { ...cur[section], [field]: value } };
      return next;
    });
    setDirty(true);
  };

  if (err) {
    return (
      <div className="flex h-full items-center justify-center text-sm text-destructive">
        <AlertCircle className="mr-2 h-4 w-4" /> {err}
      </div>
    );
  }
  if (!schema || !config) {
    return (
      <div className="flex h-full items-center justify-center text-sm text-muted-foreground">
        <Loader2 className="mr-2 h-4 w-4 animate-spin" /> loading…
      </div>
    );
  }

  const sections = expandSections(schema, config);

  return (
    <div className="flex h-full flex-col">
      <div className="flex items-center justify-between border-b border-border px-4 py-2">
        <div className="text-sm font-semibold">configuration</div>
        <div className="flex gap-2">
          <button
            type="button"
            onClick={refresh}
            className="rounded-md border border-border px-3 py-1 text-xs hover:bg-accent"
          >
            <RefreshCcw className="mr-1 inline h-3 w-3" /> reload
          </button>
          <button
            type="button"
            onClick={save}
            disabled={!dirty || saving}
            className={cn(
              'rounded-md bg-primary px-3 py-1 text-xs text-primary-foreground',
              (!dirty || saving) && 'opacity-40',
            )}
          >
            <Save className="mr-1 inline h-3 w-3" />
            {saving ? 'saving…' : 'save'}
          </button>
        </div>
      </div>
      <div className="flex-1 overflow-y-auto p-4">
        <div className="mx-auto max-w-3xl space-y-6">
          {sections.map(({ section, props, values }) => (
            <section
              key={section}
              className="rounded-lg border border-border bg-card"
            >
              <header className="border-b border-border px-4 py-2 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                {section}
              </header>
              <div className="space-y-3 p-4">
                {Object.entries(props).map(([field, prop]) => (
                  <FieldRow
                    key={field}
                    section={section}
                    field={field}
                    prop={prop}
                    value={values[field]}
                    onChange={(v) => onChange(section, field, v)}
                    catalog={catalog}
                    onRefreshCatalog={refreshCatalog}
                    refreshingCatalog={refreshingCatalog}
                  />
                ))}
              </div>
            </section>
          ))}
        </div>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------- helpers ---

interface ResolvedSection {
  section: string;
  props: Record<string, SchemaProp>;
  values: Record<string, unknown>;
}

function expandSections(
  schema: Schema,
  config: Record<string, Record<string, unknown>>,
): ResolvedSection[] {
  // Pydantic emits a top-level schema with $defs for each sub-model. We need
  // to follow $ref to actually see the per-section field props. The schema
  // root's `properties` has one entry per section pointing at $ref.
  const defs = (schema['$defs'] as Record<string, Schema>) ?? {};
  const rootProps = (schema.properties as Record<string, Schema>) ?? {};
  const sections: ResolvedSection[] = [];
  for (const [name, refOrInline] of Object.entries(rootProps)) {
    let propsSchema: Schema | undefined;
    const ref = (refOrInline as Record<string, string>)['$ref'];
    if (ref) {
      const key = ref.replace(/^#\/\$defs\//, '');
      propsSchema = defs[key];
    } else if ((refOrInline as Schema).properties) {
      propsSchema = refOrInline as Schema;
    }
    if (!propsSchema) continue;
    const props = (propsSchema.properties as Record<string, SchemaProp>) ?? {};
    sections.push({
      section: name,
      props,
      values: config[name] ?? {},
    });
  }
  return sections;
}

/**
 * Heuristic for "this field holds a model slug". The model dropdown takes
 * over for these so the user doesn't have to remember exact slugs.
 * - Every field in the ``models`` section qualifies.
 * - Any field whose name ends with ``_model`` qualifies (catches the
 *   per-rubric overrides in ``eval``).
 */
function isModelField(section: string, field: string): boolean {
  if (section === 'models') return true;
  return field.endsWith('_model');
}

function FieldRow({
  section,
  field,
  prop,
  value,
  onChange,
  catalog,
  onRefreshCatalog,
  refreshingCatalog,
}: {
  section: string;
  field: string;
  prop: SchemaProp;
  value: unknown;
  onChange: (v: unknown) => void;
  catalog: ModelCatalog | null;
  onRefreshCatalog: () => void;
  refreshingCatalog: boolean;
}): JSX.Element {
  const label = prop.title || field;
  const isLive = prop.live === true;
  const isModel = isModelField(section, field);

  return (
    <div className="grid grid-cols-[200px_1fr] gap-3">
      <div>
        <label className="text-xs font-medium" htmlFor={`${section}.${field}`}>
          {label}
        </label>
        <div className="text-[10px] text-muted-foreground">
          {isLive ? 'hot-reload' : 'restart'} · {prop.type || 'auto'}
        </div>
      </div>
      <div className="flex items-center gap-2">
        {isModel ? (
          <ModelPicker
            id={`${section}.${field}`}
            value={typeof value === 'string' ? value : ''}
            allowNull={prop.type !== 'string'}
            catalog={catalog}
            refreshing={refreshingCatalog}
            onRefresh={onRefreshCatalog}
            onChange={(v) => onChange(v)}
          />
        ) : (
          <FieldInput
            id={`${section}.${field}`}
            prop={prop}
            value={value}
            onChange={onChange}
          />
        )}
        {isModel && typeof value === 'string' && value && (
          <ModelHealthButton slug={value} />
        )}
      </div>
    </div>
  );
}

function FieldInput({
  id,
  prop,
  value,
  onChange,
}: {
  id: string;
  prop: SchemaProp;
  value: unknown;
  onChange: (v: unknown) => void;
}): JSX.Element {
  if (prop.enum) {
    return (
      <select
        id={id}
        value={String(value ?? '')}
        onChange={(e) => onChange(e.target.value)}
        className="flex-1 rounded-md border border-input bg-background px-2 py-1 text-sm outline-none focus:ring-2 focus:ring-ring"
      >
        {prop.enum.map((opt, i) => (
          <option key={i} value={String(opt)}>
            {String(opt)}
          </option>
        ))}
      </select>
    );
  }
  if (prop.type === 'boolean') {
    return (
      <input
        id={id}
        type="checkbox"
        checked={Boolean(value)}
        onChange={(e) => onChange(e.target.checked)}
        className="h-4 w-4"
      />
    );
  }
  if (prop.type === 'integer' || prop.type === 'number') {
    return (
      <input
        id={id}
        type="number"
        value={value === null || value === undefined ? '' : Number(value)}
        onChange={(e) => onChange(e.target.value === '' ? null : Number(e.target.value))}
        className="flex-1 rounded-md border border-input bg-background px-2 py-1 text-sm outline-none focus:ring-2 focus:ring-ring"
      />
    );
  }
  if (prop.type === 'array') {
    return (
      <input
        id={id}
        type="text"
        value={Array.isArray(value) ? value.join(', ') : ''}
        onChange={(e) =>
          onChange(
            e.target.value
              .split(',')
              .map((s) => s.trim())
              .filter(Boolean),
          )
        }
        placeholder="comma,separated"
        className="flex-1 rounded-md border border-input bg-background px-2 py-1 text-sm outline-none focus:ring-2 focus:ring-ring"
      />
    );
  }
  return (
    <input
      id={id}
      type="text"
      value={value === null || value === undefined ? '' : String(value)}
      onChange={(e) => onChange(e.target.value)}
      className="flex-1 rounded-md border border-input bg-background px-2 py-1 text-sm font-mono outline-none focus:ring-2 focus:ring-ring"
    />
  );
}

// Sentinel value the <select> uses when the current slug doesn't appear in
// either the OpenRouter or Ollama list. Picking it flips the row into
// free-text mode so power users can still type any slug.
const CUSTOM_SENTINEL = '__custom__';
// Sentinel for "(unset)" — only available for nullable model fields like
// ``eval.outcome_judge_model``.
const NULL_SENTINEL = '__null__';

function ModelPicker({
  id,
  value,
  allowNull,
  catalog,
  refreshing,
  onRefresh,
  onChange,
}: {
  id: string;
  value: string;
  allowNull: boolean;
  catalog: ModelCatalog | null;
  refreshing: boolean;
  onRefresh: () => void;
  onChange: (v: string | null) => void;
}): JSX.Element {
  const orList = catalog?.openrouter ?? [];
  // Normalize Ollama names to ``ollama/<name>`` so the slug the user
  // saves matches what ``shared.get_llm`` expects.
  const ollamaList = useMemo(
    () => (catalog?.ollama ?? []).map((n) => `ollama/${n}`),
    [catalog?.ollama],
  );

  const knownSlugs = useMemo(
    () => new Set([...orList, ...ollamaList]),
    [orList, ollamaList],
  );

  // "Custom" mode is sticky: once the user picks "custom" they keep the
  // text input even after typing a slug that happens to match a known
  // one, so the dropdown doesn't yank focus out of their field.
  const [customMode, setCustomMode] = useState(
    () => Boolean(value) && !knownSlugs.has(value),
  );

  // If the catalog loads after the user picks an unknown slug, flip into
  // custom mode automatically so they can see + edit what they entered.
  useEffect(() => {
    if (!catalog) return;
    if (value && !knownSlugs.has(value) && !customMode) {
      setCustomMode(true);
    }
  }, [catalog, value, knownSlugs, customMode]);

  if (!catalog) {
    return (
      <div className="flex flex-1 items-center gap-2">
        <input
          id={id}
          type="text"
          value={value}
          onChange={(e) => onChange(e.target.value)}
          className="flex-1 rounded-md border border-input bg-background px-2 py-1 text-sm font-mono outline-none focus:ring-2 focus:ring-ring"
          placeholder="loading model catalog…"
        />
      </div>
    );
  }

  const selectValue = customMode
    ? CUSTOM_SENTINEL
    : !value && allowNull
      ? NULL_SENTINEL
      : value || CUSTOM_SENTINEL;

  return (
    <div className="flex flex-1 items-center gap-2">
      <select
        id={id}
        value={selectValue}
        onChange={(e) => {
          const v = e.target.value;
          if (v === CUSTOM_SENTINEL) {
            setCustomMode(true);
            return;
          }
          setCustomMode(false);
          if (v === NULL_SENTINEL) {
            onChange(null);
          } else {
            onChange(v);
          }
        }}
        className="flex-1 rounded-md border border-input bg-background px-2 py-1 text-sm font-mono outline-none focus:ring-2 focus:ring-ring"
      >
        {allowNull && (
          <option value={NULL_SENTINEL}>(use default)</option>
        )}
        <optgroup label="OpenRouter">
          {orList.map((s) => (
            <option key={s} value={s}>
              {s}
            </option>
          ))}
        </optgroup>
        <optgroup
          label={
            catalog.ollama_available
              ? `Ollama · ${catalog.ollama_host}${ollamaList.length === 0 ? ' (no models pulled)' : ''}`
              : `Ollama · ${catalog.ollama_host} (unreachable)`
          }
        >
          {ollamaList.length === 0 ? (
            <option value="" disabled>
              {catalog.ollama_available
                ? 'no models pulled — run `ollama pull <name>`'
                : catalog.ollama_error || 'ollama unreachable'}
            </option>
          ) : (
            ollamaList.map((s) => (
              <option key={s} value={s}>
                {s}
              </option>
            ))
          )}
        </optgroup>
        <option value={CUSTOM_SENTINEL}>custom…</option>
      </select>
      {customMode && (
        <input
          type="text"
          value={value}
          onChange={(e) => onChange(e.target.value)}
          placeholder="provider/model"
          className="flex-1 rounded-md border border-input bg-background px-2 py-1 text-sm font-mono outline-none focus:ring-2 focus:ring-ring"
        />
      )}
      <button
        type="button"
        onClick={onRefresh}
        disabled={refreshing}
        title="refresh ollama tags + openrouter catalog"
        className="rounded-md border border-border px-2 py-1 text-xs text-muted-foreground hover:bg-accent disabled:opacity-40"
      >
        {refreshing ? (
          <Loader2 className="h-3 w-3 animate-spin" />
        ) : (
          <RefreshCcw className="h-3 w-3" />
        )}
      </button>
    </div>
  );
}

function ModelHealthButton({ slug }: { slug: string }): JSX.Element {
  const [state, setState] = useState<'idle' | 'checking' | 'ok' | 'err'>('idle');
  const [msg, setMsg] = useState('');
  const check = async () => {
    setState('checking');
    setMsg('');
    try {
      const r = await api.modelHealth(slug);
      if (r.ok) {
        setState('ok');
        setMsg(`${r.latency_ms}ms · ${r.provider}`);
      } else {
        setState('err');
        setMsg(r.error || 'failed');
      }
    } catch (e) {
      setState('err');
      setMsg(e instanceof Error ? e.message : String(e));
    }
  };
  return (
    <button
      type="button"
      onClick={check}
      title={msg || 'verify model slug'}
      className={cn(
        'rounded-md border px-2 py-1 text-xs',
        state === 'ok' && 'border-emerald-500/40 text-emerald-300',
        state === 'err' && 'border-destructive/40 text-destructive',
        state === 'idle' && 'border-border text-muted-foreground hover:bg-accent',
      )}
    >
      {state === 'checking' && <Loader2 className="h-3 w-3 animate-spin" />}
      {state === 'ok' && <CheckCircle2 className="h-3 w-3" />}
      {state === 'err' && <AlertCircle className="h-3 w-3" />}
      {state === 'idle' && 'verify'}
    </button>
  );
}
