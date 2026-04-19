const state = {
  currentTokens: 0,
  maxTokens: 32768,
  messages: [],
  graphNodes: [],
  graphEdges: [],
  graphTransform: {
    scale: 1,
    x: 0,
    y: 0
  },
  drag: null,
  bridgeMode: "loading"
};

function createHttpBridge(baseUrl = "http://127.0.0.1:4318") {
  return {
    async bootstrapState() {
      const [health, graph] = await Promise.all([
        fetchJson(`${baseUrl}/v1/health`),
        fetchJson(`${baseUrl}/v1/conversations/demo-thread/graph`)
      ]);
      return {
        sidecar: {
          status: health.status,
          preferredDevice: health.modelDevice,
          orchestrationDevice: health.orchestrationDevice
        },
        conversation: graph.conversation,
        graph: graph.graph,
        messages: graph.messages
      };
    },
    async sendMessage(text, currentTokens) {
      return await fetchJson(`${baseUrl}/v1/chat/respond`, {
        method: "POST",
        headers: {
          "content-type": "application/json"
        },
        body: JSON.stringify({
          conversationId: "demo-thread",
          userText: text,
          activeWindow: {
            maxTokens: state.maxTokens,
            currentTokens
          }
        })
      });
    },
    async resetConversation() {
      return await fetchJson(`${baseUrl}/v1/conversations/demo-thread/reset`, {
        method: "POST",
        headers: {
          "content-type": "application/json"
        },
        body: JSON.stringify({
          conversationId: "demo-thread"
        })
      });
    }
  };
}

function createUnavailableBridge() {
  return {
    async bootstrapState() {
      return {
        sidecar: {
          status: "offline",
          preferredDevice: "mps",
          orchestrationDevice: "cpu"
        },
        conversation: {
          id: "offline",
          title: "offline",
          currentTokens: 0,
          maxTokens: 32768
        },
        graph: {
          nodes: [],
          edges: []
        },
        messages: [
          {
            id: "offline-message",
            role: "system",
            roleLabel: "System",
            text: "Local sidecar is not running on port 4318."
          }
        ]
      };
    },
    async sendMessage() {
      return { ok: false, error: "sidecar_unavailable" };
    },
    async resetConversation() {
      return await this.bootstrapState();
    }
  };
}

async function fetchJson(url, options) {
  const response = await fetch(url, options);
  if (!response.ok) {
    const detail = await response.text();
    throw new Error(detail || `http_${response.status}`);
  }
  return await response.json();
}

const bridge = window.conChat ?? createHttpBridge();

function byId(id) {
  return document.getElementById(id);
}

function escapeHtml(text) {
  return String(text)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}

function applyGraphTransform(group) {
  group.setAttribute(
    "transform",
    `translate(${state.graphTransform.x} ${state.graphTransform.y}) scale(${state.graphTransform.scale})`
  );
}

function drawGraph() {
  const svg = byId("memory-graph");
  const namespace = "http://www.w3.org/2000/svg";
  svg.replaceChildren();

  const group = document.createElementNS(namespace, "g");
  group.setAttribute("id", "graph-layer");
  applyGraphTransform(group);
  svg.appendChild(group);

  const index = new Map(state.graphNodes.map((node) => [node.id, node]));

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
      type === "recalled-into" ? "rgba(243, 186, 103, 0.7)" : "rgba(127, 217, 228, 0.38)"
    );
    line.setAttribute("stroke-width", type === "recalled-into" ? "4" : "3");
    line.setAttribute("stroke-linecap", "round");
    group.appendChild(line);
  });

  state.graphNodes.forEach((node) => {
    const circle = document.createElementNS(namespace, "circle");
    circle.setAttribute("cx", node.x);
    circle.setAttribute("cy", node.y);
    circle.setAttribute("r", node.type === "active" ? "18" : "14");
    circle.setAttribute(
      "fill",
      node.type === "memory"
        ? "#f3ba67"
        : node.type === "recalled"
          ? "#7fd9e4"
          : node.type === "active"
            ? "#6fd48d"
            : "#a9b8ce"
    );
    circle.setAttribute("data-node-id", node.id);
    circle.setAttribute("data-node-type", node.type);
    group.appendChild(circle);

    const label = document.createElementNS(namespace, "text");
    label.setAttribute("x", node.x + 24);
    label.setAttribute("y", node.y + 5);
    label.setAttribute("fill", "#111318");
    label.setAttribute("font-size", "11");
    label.setAttribute("font-family", "\"Avenir Next\", Inter, system-ui, sans-serif");
    label.textContent = node.label;
    label.setAttribute("data-node-id", node.id);
    label.setAttribute("data-node-type", node.type);
    const labelTitle = document.createElementNS(namespace, "title");
    labelTitle.textContent = `${node.label} · ${node.type}`;
    label.appendChild(labelTitle);
    group.appendChild(label);

    const title = document.createElementNS(namespace, "title");
    title.textContent = `${node.label} · ${node.type}`;
    circle.appendChild(title);
  });
}

function updateGraphTransform() {
  const group = byId("graph-layer");
  if (group) {
    applyGraphTransform(group);
  }
}

function setBudget(current, max) {
  state.currentTokens = current;
  state.maxTokens = max;
}

function renderMessages(messages) {
  const thread = byId("thread");
  thread.replaceChildren();

  messages
    .filter((message) => message.role !== "system")
    .forEach((message) => {
    const article = document.createElement("article");
    article.className = `message ${message.role}${message.pending ? " pending" : ""}`;

    const role = document.createElement("div");
    role.className = "message-role";
    role.textContent = message.roleLabel;

    const body = document.createElement("p");
    body.textContent = message.text;

    const actions = document.createElement("div");
    actions.className = "message-actions";

    const copy = document.createElement("button");
    copy.type = "button";
    copy.className = "copy-button";
    copy.dataset.copy = message.text;
    copy.textContent = "Copy";

    actions.appendChild(copy);
    article.appendChild(role);
    article.appendChild(body);
    article.appendChild(actions);
    thread.appendChild(article);
    });

  thread.scrollTop = thread.scrollHeight;
}

function applyBootstrap(initial) {
  state.bridgeMode = initial.sidecar?.status || "unknown";
  state.messages = initial.messages;
  state.graphNodes = initial.graph.nodes;
  state.graphEdges = initial.graph.edges;
  setBudget(initial.conversation.currentTokens, initial.conversation.maxTokens);
  renderMessages(state.messages);
  drawGraph();
}

function bindComposer() {
  const input = byId("composer-input");

  input.addEventListener("input", () => {
    input.style.height = "auto";
    input.style.height = `${Math.min(input.scrollHeight, 160)}px`;
  });

  input.addEventListener("keydown", async (event) => {
    if (event.key !== "Enter" || event.shiftKey) {
      return;
    }
    event.preventDefault();
    byId("composer").requestSubmit();
  });

  byId("composer").addEventListener("submit", async (event) => {
    event.preventDefault();
    const text = input.value.trim();
    if (!text) {
      return;
    }

    const pendingMessage = {
      id: `pending-${Date.now()}`,
      role: "user",
      roleLabel: "You",
      text,
      pending: true
    };
    state.messages = [...state.messages, pendingMessage];
    renderMessages(state.messages);

    input.value = "";
    input.style.height = "42px";

    let result;
    try {
      result = await bridge.sendMessage(text, state.currentTokens);
    } catch (_error) {
      state.messages = state.messages.filter((message) => message.id !== pendingMessage.id);
      renderMessages(state.messages);
      return;
    }

    state.messages = result.messages;
    setBudget(result.conversation.currentTokens, result.conversation.maxTokens);
    state.graphNodes = result.graph.nodes;
    state.graphEdges = result.graph.edges;
    renderMessages(state.messages);
    drawGraph();
  });

  byId("thread").addEventListener("click", async (event) => {
    const button = event.target.closest(".copy-button");
    if (!button) {
      return;
    }
    const text = button.getAttribute("data-copy") || "";
    try {
      await navigator.clipboard.writeText(text);
      button.textContent = "Copied";
      window.setTimeout(() => {
        button.textContent = "Copy";
      }, 1200);
    } catch (_error) {
      button.textContent = "Failed";
      window.setTimeout(() => {
        button.textContent = "Copy";
      }, 1200);
    }
  });
}

function bindGraphControls() {
  const stage = byId("graph-stage");

  stage.addEventListener("wheel", (event) => {
    event.preventDefault();
    const delta = event.deltaY > 0 ? 0.9 : 1.1;
    state.graphTransform.scale = Math.max(0.65, Math.min(2.4, state.graphTransform.scale * delta));
    updateGraphTransform();
  });

  stage.addEventListener("pointerdown", (event) => {
    state.drag = {
      x: event.clientX,
      y: event.clientY,
      originX: state.graphTransform.x,
      originY: state.graphTransform.y
    };
    stage.setPointerCapture(event.pointerId);
  });

  stage.addEventListener("pointermove", (event) => {
    if (!state.drag) {
      return;
    }
    state.graphTransform.x = state.drag.originX + (event.clientX - state.drag.x);
    state.graphTransform.y = state.drag.originY + (event.clientY - state.drag.y);
    updateGraphTransform();
  });

  const endDrag = () => {
    state.drag = null;
  };
  stage.addEventListener("pointerup", endDrag);
  stage.addEventListener("pointercancel", endDrag);

  byId("conversation-reset").addEventListener("click", async () => {
    let result;
    try {
      result = await bridge.resetConversation();
    } catch (_error) {
      return;
    }
    state.graphTransform = { scale: 1, x: 0, y: 0 };
    state.messages = result.messages;
    setBudget(result.conversation.currentTokens, result.conversation.maxTokens);
    state.graphNodes = result.graph.nodes;
    state.graphEdges = result.graph.edges;
    renderMessages(state.messages);
    drawGraph();
  });
}

async function bootstrap() {
  let initial;
  try {
    initial = await bridge.bootstrapState();
  } catch (_error) {
    initial = await createUnavailableBridge().bootstrapState();
  }
  applyBootstrap(initial);
}

bootstrap();
bindComposer();
bindGraphControls();
