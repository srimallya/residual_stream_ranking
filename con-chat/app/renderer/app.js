const state = {
  railCollapsed: false,
  currentTokens: 0,
  maxTokens: 32768,
  graphNodes: [],
  graphEdges: [],
  selectedMemory: null
};

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

function applyBootstrap(initial) {
  byId("sidecar-status").textContent = initial.sidecar.status.replaceAll("-", " ");
  byId("device-status").textContent = `Gemma on ${initial.sidecar.preferredDevice}, harness on ${initial.sidecar.orchestrationDevice}`;
  state.graphNodes = initial.graph.nodes;
  state.graphEdges = initial.graph.edges;
  setBudget(initial.conversation.currentTokens, initial.conversation.maxTokens);
  renderMessages(initial.messages);
  renderSelectedMemory(initial.selectedMemory);
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
    const result = await window.conChat.sendMessage(text, state.currentTokens);
    if (!result.ok) {
      return;
    }

    setBudget(result.conversation.currentTokens, result.conversation.maxTokens);
    state.graphNodes = result.graph.nodes;
    state.graphEdges = result.graph.edges;
    renderMessages(result.messages);
    renderSelectedMemory(result.selectedMemory);
    drawGraph();
  });
}

async function bootstrap() {
  const initial = await window.conChat.bootstrapState();
  applyBootstrap(initial);
}

bootstrap();
bindRailToggle();
bindComposer();
