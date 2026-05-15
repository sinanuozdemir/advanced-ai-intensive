// Preload bridge.
//
// We deliberately expose a tiny surface — only what the renderer can't do
// safely via plain fetch / WebSocket against the local FastAPI server. Right
// now that's just `openExternal` for opening doc URLs in the default browser.

import { contextBridge, ipcRenderer } from 'electron';

const forgeApi = {
  openExternal: (url: string): Promise<void> =>
    ipcRenderer.invoke('forge:open-external', url),
};

contextBridge.exposeInMainWorld('forgeApi', forgeApi);

export type ForgeApi = typeof forgeApi;

declare global {
  interface Window {
    forgeApi: ForgeApi;
  }
}
