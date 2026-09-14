"use client";

import { useEffect, useState } from "react";
import { fetchDemoAccounts, login } from "@/lib/api";
import type { DemoAccount } from "@/lib/types";
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
export function Login({ onAuthenticated }: { onAuthenticated: () => void }) {
  const [accounts, setAccounts] = useState<DemoAccount[]>([]);
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
  }, []);

  async function submit(event: React.FormEvent) {
    event.preventDefault();
    setBusy(true);
    setError(null);
    try {
      await login(email, password);
      onAuthenticated();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Sign in failed.");
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
          Demonstration only. All patient data is synthetic — no record here
          describes a real person. Not medical advice.
        </p>
      </div>
    </main>
  );
}
