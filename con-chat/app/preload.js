const { contextBridge, ipcRenderer } = require("electron");

contextBridge.exposeInMainWorld("conChat", {
  bootstrapState: () => ipcRenderer.invoke("bootstrap-state"),
  sendMessage: (text, currentTokens) => ipcRenderer.invoke("send-message", { text, currentTokens })
});
