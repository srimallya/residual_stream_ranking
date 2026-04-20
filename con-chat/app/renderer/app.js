const state = {
  conversationId: "demo-thread",
  currentTokens: 0,
  maxTokens: 32768,
  messages: [],
  bootstrapReady: false,
  sending: false,
  graphNodes: [],
  graphEdges: [],
  graphTransform: {
    scale: 1,
    x: 0,
    y: 0
  },
  drag: null,
  bridgeMode: "loading",
  sidecarDevice: ""
};

function setStatusBadge(status, detail = "") {
  const node = byId("app-status");
  if (!node) {
    return;
  }
  const normalized = status || "unknown";
  const tone = normalized === "ready" ? "ready" : normalized === "offline" || normalized === "error" ? "offline" : "busy";
  node.className = `app-status ${tone}`;
  node.textContent = detail ? `${normalized} · ${detail}` : normalized;
}

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
    async sendMessageStream(text, currentTokens, handlers = {}) {
      const response = await fetch(`${baseUrl}/v1/chat/stream`, {
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
      if (!response.ok || !response.body) {
        const detail = await response.text();
        throw new Error(detail || `http_${response.status}`);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      let donePayload = null;
      let streamComplete = false;

      while (!streamComplete) {
        const { value, done } = await reader.read();
        if (done) {
          break;
        }
        buffer += decoder.decode(value, { stream: true });
        const events = buffer.split("\n\n");
        buffer = events.pop() || "";
        for (const eventBlock of events) {
          if (!eventBlock.trim()) {
            continue;
          }
          let eventName = "message";
          const dataLines = [];
          for (const line of eventBlock.split("\n")) {
            if (line.startsWith("event:")) {
              eventName = line.slice(6).trim();
            } else if (line.startsWith("data:")) {
              dataLines.push(line.slice(5).trim());
            }
          }
          const payload = JSON.parse(dataLines.join("\n") || "{}");
          if (eventName === "token") {
            handlers.onToken?.(payload);
          } else if (eventName === "done") {
            donePayload = payload;
            streamComplete = true;
            await reader.cancel();
            break;
          } else if (eventName === "error") {
            throw new Error(payload.detail || payload.error || "stream_error");
          }
        }
      }

      if (!donePayload) {
        throw new Error("stream_incomplete");
      }
      return donePayload;
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
    async sendMessageStream() {
      throw new Error("sidecar_unavailable");
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

async function copyText(text) {
  const listener = (event) => {
    event.preventDefault();
    event.clipboardData?.setData("text/plain", text);
  };

  try {
    document.addEventListener("copy", listener);
    if (document.queryCommandSupported?.("copy")) {
      const host = document.createElement("div");
      host.contentEditable = "true";
      host.setAttribute("aria-hidden", "true");
      host.style.position = "fixed";
      host.style.top = "0";
      host.style.left = "-9999px";
      host.style.opacity = "0";
      host.textContent = text;
      document.body.appendChild(host);

      const selection = window.getSelection();
      const range = document.createRange();
      range.selectNodeContents(host);
      selection?.removeAllRanges();
      selection?.addRange(range);

      const ok = document.execCommand("copy");
      selection?.removeAllRanges();
      host.remove();
      if (ok) {
        return true;
      }
    }
  } catch (_error) {
    // Fall through.
  } finally {
    document.removeEventListener("copy", listener);
  }

  if (navigator.clipboard?.writeText) {
    try {
      await navigator.clipboard.writeText(text);
      return true;
    } catch (_error) {
      // Fall through to textarea-based copy for embedded browsers that deny clipboard API.
    }
  }

  const textarea = document.createElement("textarea");
  textarea.value = text;
  textarea.setAttribute("readonly", "");
  textarea.style.position = "fixed";
  textarea.style.top = "-1000px";
  textarea.style.left = "-1000px";
  textarea.style.opacity = "0";
  document.body.appendChild(textarea);
  textarea.focus();
  textarea.select();
  textarea.setSelectionRange(0, textarea.value.length);

  let ok = false;
  try {
    ok = document.execCommand("copy");
  } catch (_error) {
    ok = false;
  } finally {
    textarea.remove();
  }
  if (ok) {
    return true;
  }

  try {
    const selection = window.getSelection();
    const range = document.createRange();
    const helper = document.createElement("div");
    helper.textContent = text;
    helper.style.position = "fixed";
    helper.style.top = "0";
    helper.style.left = "-9999px";
    document.body.appendChild(helper);
    range.selectNodeContents(helper);
    selection?.removeAllRanges();
    selection?.addRange(range);
    window.setTimeout(() => helper.remove(), 1500);
    return false;
  } catch (_error) {
    return false;
  }
}

const bridge = createHttpBridge();

function byId(id) {
  return document.getElementById(id);
}

function setComposerDisabled(disabled) {
  const input = byId("composer-input");
  const button = byId("composer-submit");
  if (!input || !button) {
    return;
  }
  input.disabled = disabled;
  button.disabled = disabled;
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

  const lines = normalized.split("\n");
  const blocks = [];
  let index = 0;

  const consumeParagraph = () => {
    const content = [];
    while (index < lines.length && lines[index].trim()) {
      const line = lines[index];
      if (
        /^#{1,6}\s+/.test(line) ||
        /^(?:- |\* )/.test(line) ||
        /^\d+\.\s+/.test(line) ||
        line.trim() === "```"
      ) {
        break;
      }
      content.push(line.trimEnd());
      index += 1;
    }
    if (content.length) {
      blocks.push(`<p>${content.map((part) => renderInlineMarkdown(part)).join("<br>")}</p>`);
    }
  };

  while (index < lines.length) {
    const rawLine = lines[index];
    const line = rawLine.trim();

    if (!line) {
      index += 1;
      continue;
    }

    if (line.startsWith("```")) {
      index += 1;
      const codeLines = [];
      while (index < lines.length && !lines[index].trim().startsWith("```")) {
        codeLines.push(lines[index]);
        index += 1;
      }
      if (index < lines.length) {
        index += 1;
      }
      blocks.push(`<pre><code>${escapeHtml(codeLines.join("\n"))}</code></pre>`);
      continue;
    }

    const headingMatch = line.match(/^(#{1,6})\s+(.+)$/);
    if (headingMatch) {
      const level = Math.min(headingMatch[1].length, 6);
      blocks.push(`<h${level}>${renderInlineMarkdown(headingMatch[2].trim())}</h${level}>`);
      index += 1;
      continue;
    }

    if (/^(?:- |\* )/.test(line)) {
      const items = [];
      while (index < lines.length && /^(?:- |\* )/.test(lines[index].trim())) {
        items.push(`<li>${renderInlineMarkdown(lines[index].trim().slice(2).trim())}</li>`);
        index += 1;
      }
      blocks.push(`<ul>${items.join("")}</ul>`);
      continue;
    }

    if (/^\d+\.\s+/.test(line)) {
      const items = [];
      while (index < lines.length && /^\d+\.\s+/.test(lines[index].trim())) {
        items.push(`<li>${renderInlineMarkdown(lines[index].trim().replace(/^\d+\.\s+/, "").trim())}</li>`);
        index += 1;
      }
      blocks.push(`<ol>${items.join("")}</ol>`);
      continue;
    }

    if (line === "---") {
      blocks.push("<hr>");
      index += 1;
      continue;
    }

    consumeParagraph();
  }

  return blocks.join("");
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
      const thinkingMeta = document.createElement("div");
      thinkingMeta.className = "thinking-meta";
      thinkingMeta.textContent = message.thinkingLabel || "thinking";
      thinkingBubble.appendChild(thinkingMeta);
      if (message.thinkingText) {
        const thinkingText = document.createElement("div");
        thinkingText.className = "thinking-text";
        thinkingText.textContent = message.thinkingText;
        thinkingBubble.appendChild(thinkingText);
      }
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
  state.sidecarDevice = initial.sidecar?.preferredDevice || "";
  state.messages = initial.messages;
  state.graphNodes = initial.graph.nodes;
  state.graphEdges = initial.graph.edges;
  setBudget(initial.conversation.currentTokens, initial.conversation.maxTokens);
  setStatusBadge(state.bridgeMode, state.sidecarDevice);
  state.bootstrapReady = true;
  setComposerDisabled(false);
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
    if (!state.bootstrapReady || state.sending) {
      return;
    }
    const text = input.value.trim();
    if (!text) {
      return;
    }

    state.sending = true;
    setStatusBadge("busy", "generating");
    setComposerDisabled(true);
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
      thinkingLabel: "thinking",
      thinkingText: ""
    };
    state.messages = [...state.messages, pendingMessage, pendingAssistant];
    renderMessages(state.messages);

    input.value = "";
    input.style.height = "42px";

    let result;
    try {
      result = await bridge.sendMessageStream(text, state.currentTokens, {
        onToken(payload) {
          state.messages = state.messages.map((message) => {
            if (message.id !== pendingAssistant.id) {
              return message;
            }
            return {
              ...message,
              text: `${message.text || ""}${payload.delta || ""}`,
              thinking: Boolean(payload.thinkingActive),
              thinkingText: `${message.thinkingText || ""}${payload.thinkingDelta || ""}`,
              pending: true
            };
          });
          renderMessages(state.messages);
        }
      });
    } catch (_error) {
      state.messages = state.messages.map((message) => {
        if (message.id === pendingMessage.id) {
          return { ...message, pending: false, failed: true };
        }
        if (message.id === pendingAssistant.id) {
          return {
            ...message,
            pending: false,
            thinking: false,
            text: "Message failed to send. Check the local sidecar and try again.",
            failed: true
          };
        }
        return message;
      });
      setStatusBadge("offline", "sidecar");
      renderMessages(state.messages);
      state.sending = false;
      setComposerDisabled(false);
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
    setStatusBadge(result.sidecar?.status || "ready", result.sidecar?.preferredDevice || state.sidecarDevice);
    renderMessages(state.messages);
    drawGraph();
    state.sending = false;
    setComposerDisabled(false);
  });

  byId("thread").addEventListener("click", async (event) => {
    const button = event.target.closest(".copy-button");
    if (!button) {
      return;
    }
    const text = button.getAttribute("data-copy") || "";
    try {
      const ok = await copyText(text);
      if (!ok) {
        button.textContent = "Press Cmd+C";
        window.setTimeout(() => {
          button.textContent = "Copy";
        }, 1800);
        return;
      }
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
    if (state.sending) {
      return;
    }
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
    setStatusBadge("ready", state.sidecarDevice);
    renderMessages(state.messages);
    drawGraph();
    hideTooltip();
  });
}

async function bootstrap() {
  setComposerDisabled(true);
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
