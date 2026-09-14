"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { streamMessage } from "@/lib/api";
import type { Turn } from "@/lib/types";
import { PendingActionCard } from "./PendingActionCard";
import { TurnDetails } from "./TurnDetails";
import styles from "./Chat.module.css";

const SUGGESTIONS = [
  "When is my next appointment?",
  "What medications am I currently taking?",
  "What did my doctor say about my knee pain?",
  "How many times was my systolic blood pressure above 140 in the last six months?",
  "Summarise my last visit and tell me which medications changed.",
];

let seq = 0;
const nextId = () => `turn-${++seq}`;

export function Chat({ onSignOut }: { onSignOut: () => void }) {
  const [turns, setTurns] = useState<Turn[]>([]);
  const [draft, setDraft] = useState("");
  const [busy, setBusy] = useState(false);
  const [conversationId, setConversationId] = useState<string | null>(null);
  const [disclaimer, setDisclaimer] = useState<string | null>(null);
  const endRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [turns]);

  const patch = useCallback((id: string, update: Partial<Turn>) => {
    setTurns((current) =>
      current.map((turn) => (turn.id === id ? { ...turn, ...update } : turn)),
    );
  }, []);

  const ask = useCallback(
    async (question: string) => {
      const text = question.trim();
      if (!text || busy) return;

      const userTurn: Turn = { id: nextId(), role: "user", text };
      const replyId = nextId();
      setTurns((current) => [
        ...current,
        userTurn,
        { id: replyId, role: "assistant", text: "", streaming: true },
      ]);
      setDraft("");
      setBusy(true);

      let streamed = "";
      try {
        await streamMessage(text, conversationId, {
          onMeta: (meta) => {
            setConversationId(meta.conversation_id);
            setDisclaimer(meta.disclaimer);
          },
          onDelta: (delta) => {
            streamed += delta;
            patch(replyId, { text: streamed });
          },
          onSources: (sources) => patch(replyId, { sources }),
          onDone: (response) => {
            // `done` carries the validated answer. Guardrails run after
            // generation, so what was streamed is provisional until here —
            // `replaces_streamed_text` says whether validation changed it.
            patch(replyId, {
              text: response.replaces_streamed_text
                ? response.answer
                : streamed || response.answer,
              sources: response.sources,
              metadata: response.metadata,
              pendingAction: response.pending_action,
              streaming: false,
            });
            setConversationId(response.conversation_id);
            setDisclaimer(response.disclaimer);
          },
          onError: (detail) =>
            patch(replyId, { error: detail, streaming: false }),
        });
      } catch (err) {
        patch(replyId, {
          error:
            err instanceof Error
              ? err.message
              : "The assistant could not be reached.",
          streaming: false,
        });
      } finally {
        setBusy(false);
        // Leaving a turn marked streaming would spin its caret forever if
        // the stream ended without a `done` frame.
        patch(replyId, { streaming: false });
      }
    },
    [busy, conversationId, patch],
  );

  return (
    <div className={styles.shell}>
      <header className={styles.header}>
        <div className={styles.brand}>
          <span className={styles.mark} aria-hidden="true" />
          <span className={styles.name}>CareCopilot</span>
        </div>
        <button type="button" className={styles.signOut} onClick={onSignOut}>
          Sign out
        </button>
      </header>

      <main className={styles.thread}>
        {turns.length === 0 ? (
          <div className={styles.empty}>
            <h1 className={styles.emptyTitle}>Ask about your record</h1>
            <p className={styles.emptyLede}>
              Appointments, medications, lab results, visits, and what your
              clinicians wrote. Answers cite the record they came from.
            </p>
            <div className={styles.suggestions}>
              {SUGGESTIONS.map((suggestion) => (
                <button
                  key={suggestion}
                  type="button"
                  className={styles.suggestion}
                  onClick={() => ask(suggestion)}
                >
                  {suggestion}
                </button>
              ))}
            </div>
          </div>
        ) : (
          <ol className={styles.turns}>
            {turns.map((turn) => (
              <li
                key={turn.id}
                className={
                  turn.role === "user" ? styles.userTurn : styles.assistantTurn
                }
              >
                {turn.role === "user" ? (
                  <p className={styles.userText}>{turn.text}</p>
                ) : (
                  <div className={styles.assistantBody}>
                    {turn.error ? (
                      <p className={styles.turnError} role="alert">
                        {turn.error}
                      </p>
                    ) : (
                      <p className={styles.assistantText}>
                        {turn.text}
                        {turn.streaming && (
                          <span className={styles.caret} aria-hidden="true" />
                        )}
                      </p>
                    )}

                    {!turn.streaming && (
                      <TurnDetails
                        sources={turn.sources}
                        metadata={turn.metadata}
                      />
                    )}

                    {turn.pendingAction && !turn.actionOutcome && (
                      <PendingActionCard
                        action={turn.pendingAction}
                        onResolved={(outcome) =>
                          patch(turn.id, { actionOutcome: outcome })
                        }
                      />
                    )}
                    {turn.actionOutcome && (
                      <p className={styles.outcome}>{turn.actionOutcome}</p>
                    )}
                  </div>
                )}
              </li>
            ))}
          </ol>
        )}
        <div ref={endRef} />
      </main>

      <footer className={styles.composer}>
        <form
          className={styles.form}
          onSubmit={(event) => {
            event.preventDefault();
            ask(draft);
          }}
        >
          <textarea
            id="message"
            className={styles.input}
            value={draft}
            rows={1}
            placeholder="Ask about your record…"
            onChange={(event) => setDraft(event.target.value)}
            onKeyDown={(event) => {
              // Enter sends, Shift+Enter breaks the line — the convention
              // every chat UI uses, so not following it surprises people.
              if (event.key === "Enter" && !event.shiftKey) {
                event.preventDefault();
                ask(draft);
              }
            }}
            disabled={busy}
          />
          <button
            type="submit"
            className={styles.send}
            disabled={busy || draft.trim().length === 0}
          >
            {busy ? "…" : "Send"}
          </button>
        </form>
        <p className={styles.disclaimer}>
          {disclaimer ??
            "Demonstration with synthetic patient data. Not medical advice."}
        </p>
      </footer>
    </div>
  );
}
