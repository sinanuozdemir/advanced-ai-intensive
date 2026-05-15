import { cn } from '@/lib/cn';
import {
  MessageSquare,
  Users,
  Brain,
  Settings,
  Plug,
  Gauge,
  type LucideIcon,
} from 'lucide-react';

export type ViewId =
  | 'chat'
  | 'agents'
  | 'memory'
  | 'mcp'
  | 'eval'
  | 'settings';

interface NavItem {
  id: ViewId;
  label: string;
  icon: LucideIcon;
}

const NAV: NavItem[] = [
  { id: 'chat', label: 'Chat', icon: MessageSquare },
  { id: 'agents', label: 'Agents', icon: Users },
  { id: 'memory', label: 'Memory', icon: Brain },
  { id: 'mcp', label: 'MCP', icon: Plug },
  { id: 'eval', label: 'Eval', icon: Gauge },
  { id: 'settings', label: 'Settings', icon: Settings },
];

interface Props {
  current: ViewId;
  onChange: (next: ViewId) => void;
}

export function Sidebar({ current, onChange }: Props): JSX.Element {
  return (
    <nav className="flex w-14 flex-col items-center gap-1 border-r border-border bg-card py-3">
      <div className="mb-3 flex h-9 w-9 items-center justify-center rounded-md bg-primary/10 text-primary font-bold">
        F
      </div>
      {NAV.map((item) => {
        const Icon = item.icon;
        const active = item.id === current;
        return (
          <button
            key={item.id}
            type="button"
            onClick={() => onChange(item.id)}
            className={cn(
              'flex h-10 w-10 items-center justify-center rounded-md transition-colors',
              active
                ? 'bg-primary/15 text-primary'
                : 'text-muted-foreground hover:bg-accent hover:text-foreground',
            )}
            title={item.label}
          >
            <Icon className="h-5 w-5" />
          </button>
        );
      })}
    </nav>
  );
}
