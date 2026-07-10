const API_URL = "http://localhost:8000"; // change to your EC2 public IP + port when deployed

const messagesEl = document.getElementById("messages");
const form = document.getElementById("chatForm");
const input = document.getElementById("userInput");
const sendBtn = document.getElementById("sendBtn");
const sendIcon = document.getElementById("sendIcon");
const loadingIcon = document.getElementById("loadingIcon");

// In-memory chat history sent to the backend for query transformation context
let chatHistory = [];
let isStreaming = false;

// Render the initial welcome message
addAssistantBubble({
  content:
    "Hi, I'm your support agent. Ask me anything about your account, orders, or policies — I'll cite exactly where each answer comes from.",
});

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const message = input.value.trim();
  if (!message || isStreaming) return;

  addUserBubble(message);
  chatHistory.push({ role: "user", content: message });
  input.value = "";
  setStreaming(true);

  const assistantBubble = addAssistantBubble({ content: "", streaming: true });

  try {
    const res = await fetch(`${API_URL}/api/chat/stream`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message, chat_history: chatHistory }),
    });

    if (!res.ok || !res.body) {
      throw new Error(`Request failed with status ${res.status}`);
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    let fullText = "";

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const events = buffer.split("\n\n");
      buffer = events.pop() || "";

      for (const evt of events) {
        const line = evt.trim();
        if (!line.startsWith("data:")) continue;

        const jsonStr = line.slice(5).trim();
        if (!jsonStr) continue;

        const payload = JSON.parse(jsonStr);

        if (payload.type === "token") {
          fullText += payload.content;
          updateBubbleText(assistantBubble, fullText);
        } else if (payload.type === "done") {
          finalizeBubble(assistantBubble, {
            citations: payload.citations,
            ticketStatus: payload.ticket_status,
            escalationReason: payload.escalation_reason,
          });
          chatHistory.push({ role: "assistant", content: fullText });
        }
      }
    }
  } catch (err) {
    updateBubbleText(
      assistantBubble,
      "Something went wrong reaching the support backend. Please try again in a moment."
    );
    finalizeBubble(assistantBubble, {});
  } finally {
    setStreaming(false);
  }
});

// ---------------------------------------------------------------------------
// DOM helpers
// ---------------------------------------------------------------------------

function setStreaming(value) {
  isStreaming = value;
  input.disabled = value;
  sendBtn.disabled = value || !input.value.trim();
  sendIcon.classList.toggle("hidden", value);
  loadingIcon.classList.toggle("hidden", !value);
}

function addUserBubble(text) {
  const row = document.createElement("div");
  row.className = "msg-row user";
  row.innerHTML = `
    <div class="avatar user">🧑</div>
    <div class="bubble user"></div>
  `;
  row.querySelector(".bubble").textContent = text;
  messagesEl.appendChild(row);
  scrollToBottom();
}

function addAssistantBubble({ content, streaming }) {
  const row = document.createElement("div");
  row.className = "msg-row assistant";
  row.innerHTML = `
    <div class="avatar assistant">🤖</div>
    <div class="bubble assistant">
      <span class="bubble-text"></span>
    </div>
  `;
  const bubble = row.querySelector(".bubble");
  bubble.querySelector(".bubble-text").textContent = content || "";
  messagesEl.appendChild(row);
  scrollToBottom();
  return bubble;
}

function updateBubbleText(bubbleEl, text) {
  bubbleEl.querySelector(".bubble-text").textContent = text;
  scrollToBottom();
}

function finalizeBubble(bubbleEl, { citations = [], ticketStatus, escalationReason }) {
  if (ticketStatus) {
    const isResolved = ticketStatus === "Resolved";
    const isEscalated = ticketStatus === "Unresolved - Insufficient KB";
    const pill = document.createElement("div");
    pill.className = `status-pill ${isResolved ? "resolved" : isEscalated ? "escalated" : ""}`;
    pill.textContent = ticketStatus.toUpperCase();
    bubbleEl.appendChild(pill);
  }

  if (citations && citations.length > 0) {
    const wrap = document.createElement("div");
    wrap.className = "citations";
    wrap.innerHTML = `<div class="citations-label">Sources</div>`;
    citations.forEach((c) => {
      const chip = document.createElement("span");
      chip.className = "citation-chip";
      chip.textContent = `[Source: ${c}]`;
      wrap.appendChild(chip);
    });
    bubbleEl.appendChild(wrap);
  }

  if (escalationReason) {
    const note = document.createElement("div");
    note.className = "escalation-note";
    note.innerHTML = `<strong>Escalation logged:</strong> ${escalationReason}`;
    bubbleEl.appendChild(note);
  }

  scrollToBottom();
}

function scrollToBottom() {
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

// Enable/disable send button based on input content
input.addEventListener("input", () => {
  sendBtn.disabled = isStreaming || !input.value.trim();
});
