const state = {
  conversationId: "demo-thread",
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

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function createHttpBridge(baseUrl = "http://127.0.0.1:4318") {
  return {
    async bootstrapState() {
      const [health, graph] = await Promise.all([
        fetchJson(`${baseUrl}/v1/health`),
        fetchJson(`${baseUrl}/v1/conversations/${encodeURIComponent(state.conversationId)}/graph`)
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
          conversationId: state.conversationId,
          userText: text,
          activeWindow: {
            maxTokens: state.maxTokens,
            currentTokens
          }
        })
      });
    },
    async resetConversation() {
      return await fetchJson(`${baseUrl}/v1/conversations/${encodeURIComponent(state.conversationId)}/reset`, {
        method: "POST",
        headers: {
          "content-type": "application/json"
        },
        body: JSON.stringify({
          conversationId: state.conversationId
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

function renderInlineMarkdown(text) {
  return escapeHtml(text)
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/\*([^*]+)\*/g, "<em>$1</em>")
    .replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank" rel="noreferrer">$1</a>');
}

function renderMarkdown(text) {
  const normalized = String(text).replace(/\r\n/g, "\n").trim();
  if (!normalized) {
    return "";
  }

  const blocks = normalized.split(/\n{2,}/).map((block) => block.trim()).filter(Boolean);
  return blocks.map((block) => {
    if (block.startsWith("- ")) {
      const items = block
        .split("\n")
        .filter((line) => line.startsWith("- "))
        .map((line) => `<li>${renderInlineMarkdown(line.slice(2).trim())}</li>`)
        .join("");
      return `<ul>${items}</ul>`;
    }
    if (block.startsWith("```") && block.endsWith("```")) {
      const code = block.replace(/^```[\w-]*\n?/, "").replace(/\n?```$/, "");
      return `<pre><code>${escapeHtml(code)}</code></pre>`;
    }
    return `<p>${renderInlineMarkdown(block).replace(/\n/g, "<br>")}</p>`;
  }).join("");
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
    circle.setAttribute("r", node.type === "active" ? "18" : node.type === "entity" ? "11" : "14");
    circle.setAttribute(
      "fill",
      node.type === "memory"
        ? "#f3ba67"
        : node.type === "recalled"
          ? "#7fd9e4"
          : node.type === "active"
            ? "#6fd48d"
            : node.type === "entity"
              ? "#111318"
              : "#a9b8ce"
    );
    circle.setAttribute("data-node-id", node.id);
    circle.setAttribute("data-node-type", node.type);
    circle.setAttribute("data-node-label", node.label);
    circle.setAttribute("data-node-detail", node.detail || `${node.label} · ${node.type}`);
    group.appendChild(circle);
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

    const body = document.createElement("div");
    body.className = "message-body";
    body.innerHTML = renderMarkdown(message.text);

    if (message.thinking) {
      const thinkingWrap = document.createElement("div");
      thinkingWrap.className = `thinking-wrap${message.collapsing ? " collapsing" : ""}`;
      const thinkingBubble = document.createElement("div");
      thinkingBubble.className = "thinking-bubble";
      thinkingBubble.textContent = message.thinkingLabel || "thinking";
      thinkingWrap.appendChild(thinkingBubble);
      article.appendChild(role);
      article.appendChild(thinkingWrap);
      if (message.text) {
        article.appendChild(body);
      }
    } else {
      article.appendChild(role);
      article.appendChild(body);
    }

    const actions = document.createElement("div");
    actions.className = "message-actions";

    const copy = document.createElement("button");
    copy.type = "button";
    copy.className = "copy-button";
    copy.dataset.copy = message.text;
    copy.textContent = "Copy";

    if (!message.thinking) {
      actions.appendChild(copy);
      article.appendChild(actions);
    }
    thread.appendChild(article);
    });

  thread.scrollTop = thread.scrollHeight;
}

function applyBootstrap(initial) {
  state.conversationId = initial.conversation.id || "demo-thread";
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
    const pendingAssistant = {
      id: `thinking-${Date.now()}`,
      role: "assistant",
      roleLabel: "con-chat",
      text: "",
      pending: true,
      thinking: true,
      thinkingLabel: "thinking"
    };
    state.messages = [...state.messages, pendingMessage, pendingAssistant];
    renderMessages(state.messages);

    input.value = "";
    input.style.height = "42px";

    let result;
    try {
      result = await bridge.sendMessage(text, state.currentTokens);
    } catch (_error) {
      state.messages = state.messages.filter((message) => message.id !== pendingMessage.id && message.id !== pendingAssistant.id);
      renderMessages(state.messages);
      return;
    }

    state.messages = state.messages.map((message) => (
      message.id === pendingAssistant.id
        ? { ...message, collapsing: true }
        : message
    ));
    renderMessages(state.messages);
    await new Promise((resolve) => window.setTimeout(resolve, 180));

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
  const tooltip = byId("graph-tooltip");

  const hideTooltip = () => {
    tooltip.classList.remove("visible");
    tooltip.textContent = "";
  };

  const showTooltip = (event, target) => {
    const detail = target.getAttribute("data-node-detail");
    if (!detail) {
      hideTooltip();
      return;
    }
    tooltip.textContent = detail;
    tooltip.classList.add("visible");
    const bounds = stage.getBoundingClientRect();
    const x = clamp(event.clientX - bounds.left + 12, 8, bounds.width - 188);
    const y = clamp(event.clientY - bounds.top + 12, 8, bounds.height - 52);
    tooltip.style.transform = `translate3d(${x}px, ${y}px, 0)`;
  };

  stage.addEventListener("wheel", (event) => {
    event.preventDefault();
    const zoomFactor = Math.exp(-event.deltaY * 0.0012);
    state.graphTransform.scale = clamp(state.graphTransform.scale * zoomFactor, 0.65, 2.4);
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
  stage.addEventListener("pointerleave", () => {
    endDrag();
    hideTooltip();
  });

  stage.addEventListener("pointermove", (event) => {
    const target = event.target.closest("[data-node-id]");
    if (target && !state.drag) {
      showTooltip(event, target);
      return;
    }
    if (!state.drag) {
      hideTooltip();
    }
  });

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
    hideTooltip();
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
