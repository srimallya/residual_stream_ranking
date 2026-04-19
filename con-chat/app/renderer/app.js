const state = {
  railCollapsed: false,
  currentTokens: 0,
  maxTokens: 32768,
  graphNodes: [],
  graphEdges: [],
  selectedMemory: null,
  previewMode: false
};

const previewStore = {
  conversation: {
    id: "preview-thread",
    title: "Browser preview conversation",
    currentTokens: 21184,
    maxTokens: 32768
  },
  messages: [
    {
      id: "turn-0001",
      role: "user",
      roleLabel: "You",
      text: "Keep the conversation continuous, but stop dragging the whole past as live context."
    },
    {
      id: "turn-0002",
      role: "assistant",
      roleLabel: "con-chat",
      text: "I’ll keep the active thread bounded, seal older stable regions into replayable memory objects, and recall them only when they matter."
    },
    {
      id: "turn-0003",
      role: "system",
      roleLabel: "Memory Event",
      text: "Turns 18-25 compacted into token@34/fp16 and linked into the graph."
    }
  ],
  selectedMemory: {
    id: "memory-0001",
    kind: "token@34/fp16",
    tier: "warm",
    bytes: 1536,
    turnRange: "turn-0018..turn-0025",
    lastUsedLabel: "recently"
  },
  graph: {
    nodes: [
      { id: "turn-0001", label: "Turn 0001", type: "turn", x: 72, y: 72 },
      { id: "turn-0002", label: "Turn 0002", type: "turn", x: 72, y: 150 },
      { id: "memory-0001", label: "Memory", type: "memory", x: 196, y: 156 },
      { id: "turn-0003", label: "Active", type: "active", x: 72, y: 320 },
      { id: "recall-memory-0001", label: "Recalled", type: "recalled", x: 196, y: 364 }
    ],
    edges: [
      { from: "turn-0001", to: "turn-0002", type: "chronological" },
      { from: "turn-0002", to: "turn-0003", type: "chronological" },
      { from: "turn-0003", to: "memory-0001", type: "compressed-into" },
      { from: "memory-0001", to: "turn-0003", type: "recalled-into" }
    ]
  }
};

function createPreviewBridge() {
  return {
    async bootstrapState() {
      return {
        appName: "con-chat",
        sidecar: {
          status: "browser-preview",
          preferredDevice: "mps",
          orchestrationDevice: "cpu"
        },
        conversation: previewStore.conversation,
        graph: previewStore.graph,
        messages: previewStore.messages,
        selectedMemory: previewStore.selectedMemory,
        timings: {
          totalRoundTripSeconds: 0.42,
          generationSeconds: 0.31,
          promptAssemblySeconds: 0.06,
          persistenceSeconds: 0.05
        }
      };
    },
    async sendMessage(text, currentTokens) {
      const created = Date.now();
      previewStore.messages.push({
        id: `turn-${created}-u`,
        role: "user",
        roleLabel: "You",
        text
      });
      previewStore.messages.push({
        id: `turn-${created}-a`,
        role: "assistant",
        roleLabel: "con-chat",
        text: "Browser preview mode is active, so this reply is mocked. The shipped app keeps the same UI but routes this turn through Electron, the Python sidecar, SQLite, and resident Gemma."
      });
      previewStore.conversation.currentTokens = Math.min(
        currentTokens + Math.max(24, Math.ceil(text.length / 3)) + 48,
        previewStore.conversation.maxTokens
      );
      previewStore.graph.nodes = [
        { id: "turn-0002", label: "Turn 0002", type: "turn", x: 72, y: 72 },
        { id: "turn-0003", label: "Turn 0003", type: "turn", x: 72, y: 150 },
        { id: `turn-${created}-u`, label: "User", type: "turn", x: 72, y: 228 },
        { id: "memory-0001", label: "Memory", type: "memory", x: 196, y: 156 },
        { id: `turn-${created}-a`, label: "Active", type: "active", x: 72, y: 320 },
        { id: "recall-memory-0001", label: "Recalled", type: "recalled", x: 196, y: 364 }
      ];
      previewStore.graph.edges = [
        { from: "turn-0002", to: "turn-0003", type: "chronological" },
        { from: "turn-0003", to: `turn-${created}-u`, type: "chronological" },
        { from: `turn-${created}-u`, to: `turn-${created}-a`, type: "chronological" },
        { from: "memory-0001", to: `turn-${created}-a`, type: "recalled-into" }
      ];
      return {
        ok: true,
        assistantText: previewStore.messages.at(-1).text,
        conversation: previewStore.conversation,
        graph: previewStore.graph,
        messages: previewStore.messages,
        selectedMemory: previewStore.selectedMemory,
        timings: {
          totalRoundTripSeconds: 0.49,
          generationSeconds: 0.34,
          promptAssemblySeconds: 0.08,
          persistenceSeconds: 0.07
        }
      };
    }
  };
}

const bridge = window.conChat ?? createPreviewBridge();

function byId(id) {
  return document.getElementById(id);
}

function escapeHtml(text) {
  return String(text)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}

function drawGraph() {
  const svg = byId("memory-graph");
  const namespace = "http://www.w3.org/2000/svg";
  svg.replaceChildren();

  const index = new Map(state.graphNodes.map((node) => [node.id, node]));
  byId("graph-count").textContent = `${state.graphNodes.length} nodes`;

  state.graphEdges.forEach(({ from, to, type }) => {
    const fromNode = index.get(from);
    const toNode = index.get(to);
    if (!fromNode || !toNode) {
      return;
    }
    const line = document.createElementNS(namespace, "line");
    line.setAttribute("x1", fromNode.x);
    line.setAttribute("y1", fromNode.y);
    line.setAttribute("x2", toNode.x);
    line.setAttribute("y2", toNode.y);
    line.setAttribute(
      "stroke",
      type === "recalled-into" ? "rgba(244, 184, 96, 0.48)" : "rgba(115, 210, 222, 0.32)"
    );
    line.setAttribute("stroke-width", type === "recalled-into" ? "4" : "3");
    line.setAttribute("stroke-linecap", "round");
    svg.appendChild(line);
  });

  state.graphNodes.forEach((node) => {
    const circle = document.createElementNS(namespace, "circle");
    circle.setAttribute("cx", node.x);
    circle.setAttribute("cy", node.y);
    circle.setAttribute("r", node.type === "active" ? "18" : "14");
    circle.setAttribute(
      "fill",
      node.type === "memory"
        ? "#f4b860"
        : node.type === "recalled"
          ? "#73d2de"
          : node.type === "active"
            ? "#70d48b"
            : "#a8b7cb"
    );
    svg.appendChild(circle);

    const label = document.createElementNS(namespace, "text");
    label.setAttribute("x", node.x + 24);
    label.setAttribute("y", node.y + 4);
    label.setAttribute("fill", "#edf3fb");
    label.setAttribute("font-size", "12");
    label.textContent = node.label;
    svg.appendChild(label);
  });
}

function setBudget(current, max) {
  state.currentTokens = current;
  state.maxTokens = max;
  const pct = Math.max(0, Math.min(1, current / max));
  byId("token-budget-copy").textContent = `${current.toLocaleString()} / ${max.toLocaleString()} tokens`;
  byId("budget-fill").style.width = `${pct * 100}%`;
}

function renderMessages(messages) {
  const thread = byId("thread");
  thread.replaceChildren();

  messages.forEach((message) => {
    const article = document.createElement("article");
    article.className = `message ${message.role}`;
    article.innerHTML = `<div class="message-role">${escapeHtml(message.roleLabel)}</div><p>${escapeHtml(message.text)}</p>`;
    thread.appendChild(article);
  });

  thread.scrollTop = thread.scrollHeight;
}

function renderSelectedMemory(memory) {
  state.selectedMemory = memory;
  byId("memory-kind").textContent = memory?.kind || "none";
  byId("memory-tier").textContent = memory?.tier || "n/a";
  byId("memory-turn-range").textContent = memory?.turnRange || "n/a";
  byId("memory-bytes").textContent = memory?.bytes != null ? memory.bytes.toLocaleString() : "n/a";
  byId("memory-last-used").textContent = memory?.lastUsedLabel || "n/a";
}

function renderTimings(timings) {
  byId("timing-total").textContent =
    timings?.totalRoundTripSeconds != null ? `${timings.totalRoundTripSeconds.toFixed(2)}s` : "n/a";
  byId("timing-generation").textContent =
    timings?.generationSeconds != null ? `${timings.generationSeconds.toFixed(2)}s` : "n/a";
  byId("timing-prompt").textContent =
    timings?.promptAssemblySeconds != null ? `${timings.promptAssemblySeconds.toFixed(2)}s` : "n/a";
  byId("timing-persist").textContent =
    timings?.persistenceSeconds != null ? `${timings.persistenceSeconds.toFixed(2)}s` : "n/a";
}

function applyBootstrap(initial) {
  state.previewMode = initial.sidecar.status === "browser-preview";
  byId("preview-pill").hidden = !state.previewMode;
  byId("sidecar-status").textContent = initial.sidecar.status.replaceAll("-", " ");
  byId("device-status").textContent = `Gemma on ${initial.sidecar.preferredDevice}, harness on ${initial.sidecar.orchestrationDevice}`;
  state.graphNodes = initial.graph.nodes;
  state.graphEdges = initial.graph.edges;
  setBudget(initial.conversation.currentTokens, initial.conversation.maxTokens);
  renderMessages(initial.messages);
  renderSelectedMemory(initial.selectedMemory);
  renderTimings(initial.timings);
  drawGraph();
}

function bindRailToggle() {
  byId("rail-toggle").addEventListener("click", () => {
    state.railCollapsed = !state.railCollapsed;
    const rail = byId("memory-rail");
    rail.classList.toggle("rail-collapsed", state.railCollapsed);
    rail.classList.toggle("rail-expanded", !state.railCollapsed);
  });
}

function bindComposer() {
  byId("composer").addEventListener("submit", async (event) => {
    event.preventDefault();
    const input = byId("composer-input");
    const text = input.value.trim();
    if (!text) {
      return;
    }

    input.value = "";
    const result = await bridge.sendMessage(text, state.currentTokens);
    if (!result.ok) {
      return;
    }

    setBudget(result.conversation.currentTokens, result.conversation.maxTokens);
    state.graphNodes = result.graph.nodes;
    state.graphEdges = result.graph.edges;
    renderMessages(result.messages);
    renderSelectedMemory(result.selectedMemory);
    renderTimings(result.timings);
    drawGraph();
  });
}

async function bootstrap() {
  const initial = await bridge.bootstrapState();
  applyBootstrap(initial);
}

bootstrap();
bindRailToggle();
bindComposer();
