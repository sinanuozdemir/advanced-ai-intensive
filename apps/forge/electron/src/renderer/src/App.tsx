import { useState } from 'react';
import { Sidebar, type ViewId } from './components/Sidebar';
import { WorkspaceHeader } from './components/WorkspaceHeader';
import { ChatView } from './views/Chat';
import { AgentsView } from './views/Agents';
import { MemoryView } from './views/Memory';
import { McpView } from './views/Mcp';
import { EvalView } from './views/Eval';
import { SettingsView } from './views/Settings';

export default function App(): JSX.Element {
  const [view, setView] = useState<ViewId>('chat');
  return (
    <div className="flex h-screen w-screen overflow-hidden bg-background">
      <Sidebar current={view} onChange={setView} />
      <div className="flex flex-1 flex-col">
        <WorkspaceHeader />
        <main className="flex-1 overflow-hidden">
          {view === 'chat' && <ChatView />}
          {view === 'agents' && <AgentsView />}
          {view === 'memory' && <MemoryView />}
          {view === 'mcp' && <McpView />}
          {view === 'eval' && <EvalView />}
          {view === 'settings' && <SettingsView />}
        </main>
      </div>
    </div>
  );
}
