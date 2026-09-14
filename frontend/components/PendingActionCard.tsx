"use client";

import { useState } from "react";
import { confirmAction } from "@/lib/api";
import type { PendingAction } from "@/lib/types";
import styles from "./PendingActionCard.module.css";

/**
 * The confirm step for a proposed write (PRD §11).
 *
 * The card renders `summary` and posts `token` back untouched. It never
 * builds a request from the proposal's fields — the whole point of the
 * two-phase design is that the parameters were fixed and validated when the
 * token was signed, so a client that reconstructed them would reintroduce
 * the gap the token closes.
 *
 * "Not now" is local: there is nothing to tell the server, because nothing
 * was reserved. The proposal simply expires.
 */
export function PendingActionCard({
  action,
  onResolved,
}: {
  action: PendingAction;
  onResolved: (outcome: string) => void;
}) {
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function confirm() {
    setBusy(true);
    setError(null);
    try {
      const result = await confirmAction(action.token);
      onResolved(result.message);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not confirm.");
      setBusy(false);
    }
  }

  return (
    <div className={styles.card}>
      <div className={styles.head}>
        <span className={styles.badge}>Needs your confirmation</span>
        <span className={styles.expiry}>
          Expires {formatTime(action.expires_at)}
        </span>
      </div>

      <p className={styles.summary}>{action.summary}</p>

      {error && (
        <p className={styles.error} role="alert">
          {error}
        </p>
      )}

      <div className={styles.buttons}>
        <button
          type="button"
          className={styles.confirm}
          onClick={confirm}
          disabled={busy}
        >
          {busy ? "Confirming…" : "Yes, go ahead"}
        </button>
        <button
          type="button"
          className={styles.dismiss}
          onClick={() => onResolved("You chose not to go ahead. Nothing was changed.")}
          disabled={busy}
        >
          Not now
        </button>
      </div>
    </div>
  );
}

function formatTime(iso: string): string {
  const when = new Date(iso);
  if (Number.isNaN(when.getTime())) return "shortly";
  return when.toLocaleTimeString(undefined, {
    hour: "2-digit",
    minute: "2-digit",
  });
}
