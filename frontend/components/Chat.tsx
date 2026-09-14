"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { ApiError, streamMessage } from "@/lib/api";
import { DEMO_DISCLAIMER } from "@/lib/disclaimer";
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

export function Chat({
  onSignOut,
  onSessionExpired,
}: {
  onSignOut: () => void;
  /**
   * The token stopped being accepted mid-session — it expires after
   * JWT_TTL_MINUTES, so this is a tab left open over lunch, not an attack.
   * Handled by the parent because the remedy is to show the sign-in screen,
   * which this component cannot do from inside itself.
   */
  onSessionExpired: () => void;
}) {
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
      // Whether the turn reached a conclusion of its own. A stream can end
      // without either: the connection simply stops after `meta`, which is
      // what an unhandled exception inside the agent looks like from here.
      let concluded = false;
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
            concluded = true;
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
          onError: (detail) => {
            concluded = true;
            patch(replyId, { error: detail, streaming: false });
          },
        });

        // The stream ended cleanly but said nothing — no deltas, no `done`,
        // no `error`. Measured with the graph database stopped: an
        // exception inside the agent ended the response after its `meta`
        // frame, and the turn rendered as an empty bubble with no text and
        // no explanation. A silent failure is the one kind a patient cannot
        // act on, so it is named here rather than left blank. Partial text
        // is kept instead — half an answer is still an answer, and saying
        // it was cut short is better than discarding it.
        if (!concluded && !streamed) {
          patch(replyId, {
            error:
              "The answer was interrupted before it arrived. Please try again.",
            streaming: false,
          });
        }
      } catch (err) {
        // An expired session is not a failed answer: reporting it in the
        // thread would leave the user reading "Unauthorized" next to a
        // composer that can never succeed again. The parent sends them back
        // to sign in instead.
        if (err instanceof ApiError && err.status === 401) {
          onSessionExpired();
          return;
        }
        // Only an ApiError carries wording meant for a reader — it holds the
        // backend's own `detail`. Anything else is plumbing: a dropped
        // connection throws `TypeError: Failed to fetch`, and since that is
        // an Error too, showing `err.message` for every failure put that
        // string in front of patients and left this fallback unreachable.
        patch(replyId, {
          error:
            err instanceof ApiError
              ? err.message
              : "The assistant could not be reached. Check your connection and try again.",
          streaming: false,
        });
      } finally {
        setBusy(false);
        // Leaving a turn marked streaming would spin its caret forever if
        // the stream ended without a `done` frame.
        patch(replyId, { streaming: false });
      }
    },
    [busy, conversationId, onSessionExpired, patch],
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
        <p className={styles.disclaimer}>{disclaimer ?? DEMO_DISCLAIMER}</p>
      </footer>
    </div>
  );
}
