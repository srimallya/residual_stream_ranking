const fs = require("node:fs");
const path = require("node:path");
const { spawn } = require("node:child_process");
const { app, BrowserWindow, ipcMain } = require("electron");

const SIDECAR_PORT = 4318;
const SIDECAR_URL = `http://127.0.0.1:${SIDECAR_PORT}`;
const APP_STATE = {
  appName: "con-chat",
  conversationId: "demo-thread",
  sidecar: {
    status: "starting",
    model: "google--gemma-4-E2B-it",
    preferredDevice: "mps",
    orchestrationDevice: "cpu"
  }
};

let mainWindow = null;
let sidecarProcess = null;

function sidecarPythonPath() {
  return path.resolve(__dirname, "..", "..", ".venv", "bin", "python");
}

function sidecarScriptPath() {
  return path.resolve(__dirname, "..", "sidecar", "server.py");
}

function sidecarDbPath() {
  const dir = path.join(app.getPath("userData"), "state");
  fs.mkdirSync(dir, { recursive: true });
  return path.join(dir, "con-chat.sqlite3");
}

function delay(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function getJson(pathname, { retries = 0 } = {}) {
  let lastError;
  for (let attempt = 0; attempt <= retries; attempt += 1) {
    try {
      const response = await fetch(`${SIDECAR_URL}${pathname}`);
      if (!response.ok) {
        throw new Error(`http_${response.status}`);
      }
      return await response.json();
    } catch (error) {
      lastError = error;
      if (attempt < retries) {
        await delay(250);
      }
    }
  }
  throw lastError;
}

async function postJson(pathname, body) {
  const response = await fetch(`${SIDECAR_URL}${pathname}`, {
    method: "POST",
    headers: {
      "content-type": "application/json"
    },
    body: JSON.stringify(body)
  });

  if (!response.ok) {
    const text = await response.text();
    throw new Error(`http_${response.status}:${text}`);
  }
  return await response.json();
}

async function waitForSidecar() {
  const health = await getJson("/v1/health", { retries: 40 });
  APP_STATE.sidecar.status = health.status || "ready";
  return health;
}

function startSidecar() {
  if (sidecarProcess) {
    return;
  }

  sidecarProcess = spawn(sidecarPythonPath(), [sidecarScriptPath()], {
    cwd: path.resolve(__dirname, ".."),
    env: {
      ...process.env,
      CON_CHAT_PORT: String(SIDECAR_PORT),
      CON_CHAT_DB_PATH: sidecarDbPath(),
      CON_CHAT_MODEL_NAME: APP_STATE.sidecar.model,
      CON_CHAT_MODEL_DEVICE: APP_STATE.sidecar.preferredDevice,
      CON_CHAT_ORCHESTRATION_DEVICE: APP_STATE.sidecar.orchestrationDevice
    },
    stdio: "pipe"
  });

  sidecarProcess.stdout.on("data", (chunk) => {
    process.stdout.write(`[con-chat sidecar] ${chunk}`);
  });

  sidecarProcess.stderr.on("data", (chunk) => {
    process.stderr.write(`[con-chat sidecar] ${chunk}`);
  });

  sidecarProcess.on("exit", (code, signal) => {
    APP_STATE.sidecar.status = code === 0 ? "stopped" : `exited_${code ?? signal}`;
    sidecarProcess = null;
  });
}

async function bootstrapState() {
  const [health, graph] = await Promise.all([
    getJson("/v1/health"),
    getJson(`/v1/conversations/${APP_STATE.conversationId}/graph`)
  ]);

  APP_STATE.sidecar.status = health.status || APP_STATE.sidecar.status;

  return {
    appName: APP_STATE.appName,
    sidecar: {
      ...APP_STATE.sidecar,
      dbPath: health.dbPath
    },
    conversation: graph.conversation,
    graph: graph.graph,
    messages: graph.messages,
    selectedMemory: graph.selectedMemory
  };
}

async function sendMessage(payload) {
  const text = String(payload?.text || "").trim();
  if (!text) {
    return { ok: false, error: "empty_message" };
  }

  const response = await postJson("/v1/chat/respond", {
    conversationId: APP_STATE.conversationId,
    userText: text,
    activeWindow: {
      maxTokens: 32768,
      currentTokens: payload?.currentTokens || 0
    }
  });

  return { ok: true, ...response };
}

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1440,
    height: 940,
    minWidth: 1180,
    minHeight: 760,
    backgroundColor: "#0e1116",
    title: "con-chat",
    webPreferences: {
      preload: path.join(__dirname, "preload.js")
    }
  });

  mainWindow.loadFile(path.join(__dirname, "renderer", "index.html"));
}

async function bootstrapApp() {
  startSidecar();
  await waitForSidecar();

  ipcMain.handle("bootstrap-state", async () => bootstrapState());
  ipcMain.handle("send-message", async (_event, payload) => sendMessage(payload));

  createWindow();

  app.on("activate", () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createWindow();
    }
  });
}

app.whenReady().then(() => {
  bootstrapApp().catch((error) => {
    APP_STATE.sidecar.status = "boot-failed";
    console.error("[con-chat] failed to bootstrap app", error);
    createWindow();
  });
});

app.on("before-quit", () => {
  if (sidecarProcess) {
    sidecarProcess.kill("SIGTERM");
  }
});

app.on("window-all-closed", () => {
  if (process.platform !== "darwin") {
    app.quit();
  }
});
