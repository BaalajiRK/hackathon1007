"use client";

import { useState, useRef, useEffect, FormEvent } from "react";
import { Send, Loader2, CheckCircle2, AlertTriangle, Bot, User } from "lucide-react";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

type Role = "user" | "assistant";

interface ChatTurn {
  role: Role;
  content: string;
}

interface Message extends ChatTurn {
  id: string;
  citations?: string[];
  ticketStatus?: "In Progress" | "Resolved" | "Unresolved - Insufficient KB";
  escalationReason?: string | null;
  streaming?: boolean;
}

const API_URL =
  process.env.NEXT_PUBLIC_API_URL?.replace(/\/$/, "") || "http://localhost:8000";

// ---------------------------------------------------------------------------
// Small presentational pieces
// ---------------------------------------------------------------------------

function StatusPill({ status }: { status: Message["ticketStatus"] }) {
  if (!status) return null;

  const isResolved = status === "Resolved";
  const isEscalated = status === "Unresolved - Insufficient KB";

  return (
    <span
      className={[
        "inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-[11px] font-mono font-medium tracking-wide",
        isResolved && "bg-emerald-50 text-emerald-700 border border-emerald-200",
        isEscalated && "bg-amber-50 text-amber-700 border border-amber-200",
        !isResolved && !isEscalated && "bg-slate-100 text-slate-600 border border-slate-200",
      ]
        .filter(Boolean)
        .join(" ")}
    >
      {isResolved && <CheckCircle2 className="h-3 w-3" />}
      {isEscalated && <AlertTriangle className="h-3 w-3" />}
      {status.toUpperCase()}
    </span>
  );
}

function CitationLedger({ citations }: { citations: string[] }) {
  if (!citations || citations.length === 0) return null;

  return (
    <div className="mt-3 border-t border-slate-200 pt-2">
      <p className="text-[10px] uppercase tracking-widest text-slate-400 font-mono mb-1.5">
        Sources
      </p>
      <div className="flex flex-wrap gap-1.5">
        {citations.map((c, i) => (
          <span
            key={i}
            className="inline-flex items-center rounded-md border-l-2 border-l-indigo-400 bg-slate-50 px-2 py-1 text-[11px] font-mono text-slate-600"
          >
            [Source: {c}]
          </span>
        ))}
      </div>
    </div>
  );
}

function EscalationNote({ reason }: { reason?: string | null }) {
  if (!reason) return null;
  return (
    <div className="mt-3 rounded-md bg-amber-50 border border-amber-200 px-3 py-2 text-[12px] text-amber-800">
      <span className="font-mono font-semibold">Escalation logged: </span>
      {reason}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main widget
// ---------------------------------------------------------------------------

export default function ChatWidget() {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: "welcome",
      role: "assistant",
      content:
        "Hi, I'm your support agent. Ask me anything about your account, orders, or policies — I'll cite exactly where each answer comes from.",
    },
  ]);
  const [input, setInput] = useState("");
  const [isStreaming, setIsStreaming] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
  }, [messages]);

  async function handleSubmit(e: FormEvent) {
    e.preventDefault();
    const trimmed = input.trim();
    if (!trimmed || isStreaming) return;

    const userMessage: Message = { id: crypto.randomUUID(), role: "user", content: trimmed };
    const assistantId = crypto.randomUUID();

    const history: ChatTurn[] = messages
      .filter((m) => m.id !== "welcome")
      .map((m) => ({ role: m.role, content: m.content }));

    setMessages((prev) => [
      ...prev,
      userMessage,
      { id: assistantId, role: "assistant", content: "", streaming: true },
    ]);
    setInput("");
    setIsStreaming(true);

    try {
      const res = await fetch(`${API_URL}/api/chat/stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: trimmed, chat_history: history }),
      });

      if (!res.ok || !res.body) {
        throw new Error(`Request failed with status ${res.status}`);
      }

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

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
            setMessages((prev) =>
              prev.map((m) =>
                m.id === assistantId ? { ...m, content: m.content + payload.content } : m
              )
            );
          } else if (payload.type === "done") {
            setMessages((prev) =>
              prev.map((m) =>
                m.id === assistantId
                  ? {
                      ...m,
                      streaming: false,
                      citations: payload.citations,
                      ticketStatus: payload.ticket_status,
                      escalationReason: payload.escalation_reason,
                    }
                  : m
              )
            );
          }
        }
      }
    } catch (err) {
      setMessages((prev) =>
        prev.map((m) =>
          m.id === assistantId
            ? {
                ...m,
                streaming: false,
                content:
                  "Something went wrong reaching the support backend. Please try again in a moment.",
              }
            : m
        )
      );
    } finally {
      setIsStreaming(false);
    }
  }

  return (
    <div className="flex h-[640px] w-full max-w-2xl flex-col overflow-hidden rounded-2xl border border-slate-200 bg-white shadow-sm">
      {/* Header */}
      <div className="flex items-center justify-between border-b border-slate-200 bg-slate-900 px-5 py-4">
        <div className="flex items-center gap-2.5">
          <div className="flex h-8 w-8 items-center justify-center rounded-full bg-indigo-500/20">
            <Bot className="h-4 w-4 text-indigo-300" />
          </div>
          <div>
            <p className="text-sm font-semibold text-white">Support Agent</p>
            <p className="text-[11px] font-mono text-slate-400">agentic-rag · self-correcting</p>
          </div>
        </div>
      </div>

      {/* Messages */}
      <div ref={scrollRef} className="flex-1 space-y-4 overflow-y-auto px-5 py-5 bg-slate-50/50">
        {messages.map((m) => (
          <div
            key={m.id}
            className={`flex gap-2.5 ${m.role === "user" ? "flex-row-reverse" : "flex-row"}`}
          >
            <div
              className={`mt-1 flex h-7 w-7 shrink-0 items-center justify-center rounded-full ${
                m.role === "user" ? "bg-slate-200" : "bg-slate-900"
              }`}
            >
              {m.role === "user" ? (
                <User className="h-3.5 w-3.5 text-slate-600" />
              ) : (
                <Bot className="h-3.5 w-3.5 text-white" />
              )}
            </div>

            <div className={`max-w-[80%] ${m.role === "user" ? "items-end" : "items-start"}`}>
              <div
                className={`rounded-2xl px-4 py-3 text-[13.5px] leading-relaxed whitespace-pre-wrap ${
                  m.role === "user"
                    ? "bg-slate-900 text-white rounded-tr-sm"
                    : "bg-white border border-slate-200 text-slate-800 rounded-tl-sm"
                }`}
              >
                {m.content}
                {m.streaming && (
                  <Loader2 className="ml-1 inline h-3 w-3 animate-spin text-slate-400" />
                )}

                {m.role === "assistant" && !m.streaming && m.ticketStatus && (
                  <div className="mt-2">
                    <StatusPill status={m.ticketStatus} />
                  </div>
                )}

                {m.role === "assistant" && !m.streaming && (
                  <CitationLedger citations={m.citations || []} />
                )}

                {m.role === "assistant" && !m.streaming && (
                  <EscalationNote reason={m.escalationReason} />
                )}
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Input */}
      <form
        onSubmit={handleSubmit}
        className="flex items-center gap-2 border-t border-slate-200 bg-white px-4 py-3"
      >
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Describe your issue..."
          disabled={isStreaming}
          className="flex-1 rounded-full border border-slate-200 bg-slate-50 px-4 py-2.5 text-sm text-slate-800 placeholder:text-slate-400 focus:outline-none focus:ring-2 focus:ring-indigo-400/40 disabled:opacity-60"
        />
        <button
          type="submit"
          disabled={isStreaming || !input.trim()}
          className="flex h-10 w-10 shrink-0 items-center justify-center rounded-full bg-slate-900 text-white transition hover:bg-slate-700 disabled:opacity-40"
          aria-label="Send message"
        >
          {isStreaming ? (
            <Loader2 className="h-4 w-4 animate-spin" />
          ) : (
            <Send className="h-4 w-4" />
          )}
        </button>
      </form>
    </div>
  );
}
