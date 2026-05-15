import { useEffect, useState } from 'react';
import { api, type WorkspaceInfo } from '@/lib/api';
import { Folder, AlertCircle } from 'lucide-react';

// Tiny status strip the user sees at the top of every view. Shows just the
// workspace name; we deliberately do NOT render git branch or dirty state
// here — Forge is a general-purpose agent, not a coding-agent product, and
// the workspace folder is the only context the user needs in this strip.
export function WorkspaceHeader(): JSX.Element {
  const [ws, setWs] = useState<WorkspaceInfo | null>(null);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    let alive = true;
    const tick = async () => {
      try {
        const w = await api.workspace();
        if (alive) {
          setWs(w);
          setErr(null);
        }
      } catch (e) {
        if (alive) setErr(e instanceof Error ? e.message : String(e));
      }
    };
    tick();
    const id = setInterval(tick, 10_000);
    return () => {
      alive = false;
      clearInterval(id);
    };
  }, []);

  if (err) {
    return (
      <header className="flex items-center gap-2 border-b border-border px-4 py-2 text-sm text-destructive">
        <AlertCircle className="h-4 w-4" />
        <span>backend offline · is `forge serve` running?</span>
      </header>
    );
  }

  if (!ws) {
    return (
      <header className="flex items-center gap-2 border-b border-border px-4 py-2 text-sm text-muted-foreground">
        <Folder className="h-4 w-4 animate-pulse" />
        <span>loading workspace…</span>
      </header>
    );
  }

  const repoName = ws.repo_root.split('/').filter(Boolean).slice(-1)[0] || ws.repo_root;

  return (
    <header className="flex items-center gap-3 border-b border-border bg-card px-4 py-2 text-sm">
      <Folder className="h-4 w-4 text-primary" />
      <span className="font-semibold text-foreground" title={ws.repo_root}>
        {repoName}
      </span>
    </header>
  );
}
