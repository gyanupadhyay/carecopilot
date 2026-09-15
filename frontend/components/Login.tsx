"use client";

import { useEffect, useState } from "react";
import { ApiError, fetchDemoAccounts, fetchHealth, login } from "@/lib/api";
import { DEMO_DISCLAIMER } from "@/lib/disclaimer";
import type { DemoAccount, Health } from "@/lib/types";
import styles from "./Login.module.css";

/**
 * The demo accounts are listed, with passwords, on purpose.
 *
 * Every record behind them is synthetic, and the alternative — a reviewer
 * who cannot open the demo without a handoff — costs more than the
 * credentials are worth. The backend serves them from /auth/demo-accounts
 * rather than this file hard-coding them, so the list cannot drift from
 * what was actually seeded.
 */
export function Login({
  onAuthenticated,
  notice,
}: {
  onAuthenticated: () => void;
  /** Why this screen is showing, when it is showing for a reason. */
  notice?: string | null;
}) {
  const [accounts, setAccounts] = useState<DemoAccount[]>([]);
  const [health, setHealth] = useState<Health | null>(null);
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    fetchDemoAccounts()
      .then((list) => {
        setAccounts(list);
        if (list.length > 0) {
          setEmail(list[0].email);
          setPassword(list[0].password);
        }
      })
      .catch(() => {
        // Not fatal: the form still works, it just cannot pre-fill. A demo
        // list that fails to load should not block a typed login.
        setAccounts([]);
      });

    // Which model is actually answering. Asked rather than assumed: this
    // project is built around a self-hosted Qwen3-8B, and a deployment is
    // usually serving something else — so the page should say what it is
    // instead of letting the README's description stand in for it.
    fetchHealth()
      .then(setHealth)
      .catch(() => setHealth(null));
  }, []);

  async function submit(event: React.FormEvent) {
    event.preventDefault();
    setBusy(true);
    setError(null);
    try {
      await login(email, password);
      onAuthenticated();
    } catch (err) {
      // An ApiError holds the backend's own wording ("Incorrect email or
      // password."), which is exactly what belongs here. Anything else is
      // plumbing — with the API unreachable, `fetch` throws
      // `TypeError: Failed to fetch`, and because that is an Error too, the
      // old test put those three words under the password field.
      setError(
        err instanceof ApiError
          ? err.message
          : "Could not reach the server. Check your connection and try again.",
      );
    } finally {
      setBusy(false);
    }
  }

  return (
    <main className={styles.wrap}>
      <div className={styles.card}>
        <p className={styles.eyebrow}>CareCopilot</p>
        <h1 className={styles.title}>Your record, answered</h1>
        <p className={styles.lede}>
          Ask about appointments, medications, lab results and what your
          clinicians wrote. Every answer cites the record it came from.
        </p>

        {notice && (
          <p className={styles.notice} role="status">
            {notice}
          </p>
        )}

        <form onSubmit={submit} className={styles.form}>
          <label className={styles.field}>
            <span>Email</span>
            <input
              id="email"
              type="email"
              value={email}
              autoComplete="username"
              onChange={(e) => setEmail(e.target.value)}
              required
            />
          </label>
          <label className={styles.field}>
            <span>Password</span>
            <input
              id="password"
              type="password"
              value={password}
              autoComplete="current-password"
              onChange={(e) => setPassword(e.target.value)}
              required
            />
          </label>

          {error && (
            <p className={styles.error} role="alert">
              {error}
            </p>
          )}

          <button className={styles.submit} type="submit" disabled={busy}>
            {busy ? "Signing in…" : "Sign in"}
          </button>
        </form>

        {accounts.length > 0 && (
          <div className={styles.accounts}>
            <p className={styles.accountsLabel}>Demo accounts</p>
            <div className={styles.accountList}>
              {accounts.slice(0, 4).map((account) => (
                <button
                  key={account.email}
                  type="button"
                  className={styles.account}
                  onClick={() => {
                    setEmail(account.email);
                    setPassword(account.password);
                  }}
                >
                  <span className={styles.accountName}>
                    {account.display_name}
                  </span>
                  <span className={styles.accountEmail}>
                    {account.patient_external_id}
                  </span>
                </button>
              ))}
            </div>
          </div>
        )}

        <p className={styles.disclaimer}>
          {DEMO_DISCLAIMER} No record here describes a real person.
          {health && (
            <>
              {" "}
              Answers come from <code>{health.model}</code>
              {health.llm_provider === "ollama" ||
              health.llm_provider === "vllm"
                ? ", self-hosted."
                : ` via ${health.llm_provider}. Run it yourself and it serves a self-hosted model instead.`}
            </>
          )}
        </p>
      </div>
    </main>
  );
}
