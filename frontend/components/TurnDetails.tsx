"use client";

import type { ChatMetadata, Source } from "@/lib/types";
import styles from "./TurnDetails.module.css";

/**
 * The ten approved traversals, in words.
 *
 * The wire carries `kg:medications_for_condition`, which is the intent the
 * model chose and a fine thing to show a developer — but it is an identifier,
 * and the panel already has a row for identifiers. These say what the
 * traversal *did*, which is the part a reader is checking against the answer.
 */
const TRAVERSAL_LABEL: Record<string, string> = {
  conditions: "Conditions on the record",
  medications_for_condition: "Medications treating one condition",
  why_medication: "Why a medication was prescribed",
  condition_timeline: "Visits for one condition, in order",
  labs_for_condition: "Results related to one condition",
  care_team: "Clinicians seen, and their departments",
  medication_history: "Every medication on record",
  allergies: "Recorded allergies",
  procedures: "Procedures performed",
  diagnosis_history: "Diagnoses, with who recorded them",
};

const ROUTE_LABEL: Record<string, string> = {
  API: "Record lookup",
  RAG: "Clinical notes",
  // Named for what the patient gets, not for the database behind it.
  // "Knowledge graph" describes our architecture; "Connections" describes
  // the answer — which medication relates to which condition, who treated
  // what, and when.
  KG: "Connections",
  HYBRID: "Notes + record",
  TEXT_TO_SQL: "Calculated",
  ACTION: "Action",
  OUT_OF_SCOPE: "Out of scope",
};

/**
 * Citations, and the developer panel behind them (PRD §26).
 *
 * Sources are for the patient and always visible: an answer drawn from the
 * notes has to say which note. The diagnostics are collapsed because they
 * are for whoever is evaluating the system, and a patient reading about
 * reranker names learns nothing about their health.
 *
 * Nothing here is reasoning. Route, timings, token counts and the generated
 * SQL are mechanism — what the system did, not what it thought.
 */
export function TurnDetails({
  sources,
  metadata,
}: {
  sources?: Source[];
  metadata?: ChatMetadata;
}) {
  const hasSources = sources && sources.length > 0;
  if (!hasSources && !metadata) return null;

  return (
    <div className={styles.wrap}>
      {hasSources && (
        <ol className={styles.sources}>
          {sources.map((source, index) => (
            <li key={source.chunk_id ?? `${source.document_id}-${index}`}>
              <span className={styles.marker}>{index + 1}</span>
              <span className={styles.sourceBody}>
                <span className={styles.sourceTitle}>
                  {source.title ?? "Clinical note"}
                </span>
                <span className={styles.sourceMeta}>
                  {[source.section, formatDate(source.date)]
                    .filter(Boolean)
                    .join(" · ")}
                </span>
              </span>
            </li>
          ))}
        </ol>
      )}

      {metadata && (
        <details className={styles.details}>
          <summary className={styles.summary}>
            <span className={styles.route}>
              {ROUTE_LABEL[metadata.route] ?? metadata.route}
            </span>
            <span className={styles.timing}>
              {formatMs(metadata.latency_ms)}
            </span>
            {metadata.guardrails.length > 0 && (
              <span className={styles.guardrail}>
                {metadata.guardrails.length} guardrail
                {metadata.guardrails.length === 1 ? "" : "s"}
              </span>
            )}
          </summary>

          <dl className={styles.grid}>
            <Row label="Route" value={metadata.route} />
            <Row
              label="Model"
              value={
                // The resolved version when it says something the configured
                // id does not. Showing both unconditionally would print the
                // same string twice on every turn that used no alias.
                metadata.model_version && metadata.model_version !== metadata.model
                  ? `${metadata.model} → ${metadata.model_version}`
                  : metadata.model
              }
            />
            <Row label="Provider" value={metadata.llm_provider} />
            <Row
              label="Agent steps"
              value={
                metadata.agent_iterations === null
                  ? null
                  : String(metadata.agent_iterations)
              }
            />
            <Row
              label="Tools"
              value={metadata.tools_used.join(", ") || null}
            />
            <Row label="Graph traversal" value={traversalOf(metadata)} />
            <Row
              label="Structured output"
              value={
                metadata.structured_calls
                  ? `${
                      metadata.structured_calls -
                      (metadata.structured_failures ?? 0)
                    }/${metadata.structured_calls} valid`
                  : null
              }
            />
            <Row
              label="Retrieved"
              value={
                metadata.retrieved_chunks === null
                  ? null
                  : `${metadata.retrieved_chunks} candidates → ${
                      metadata.reranked_chunks ?? "?"
                    } used${
                      metadata.deduplicated_chunks
                        ? `, ${metadata.deduplicated_chunks} deduplicated`
                        : ""
                    }`
              }
            />
            <Row label="Reranker" value={metadata.reranker} />
            <Row
              // Only when it happened. A follow-up that retrieved badly
              // looks identical to a bad corpus in the chunk counts alone;
              // this is the line that separates them.
              label="Query"
              value={metadata.query_rewritten ? "rewritten for context" : null}
            />
            <Row
              label="Tokens"
              value={
                metadata.input_tokens === null && metadata.output_tokens === null
                  ? null
                  : `${metadata.input_tokens ?? 0} in / ${
                      metadata.output_tokens ?? 0
                    } out`
              }
            />
            <Row label="Cost" value={formatCost(metadata.estimated_cost_usd)} />
            <Row
              label="SQL rows"
              value={
                metadata.sql_row_count === null
                  ? null
                  : String(metadata.sql_row_count)
              }
            />
            <Row label="Action" value={metadata.action} />
            <Row
              label="Guardrails"
              value={metadata.guardrails.join(", ") || "none fired"}
            />
            <Row
              // Shown only when one occurred. A permanent "Refused: 0" row
              // trains a reader to skip the line that matters on the one
              // turn it is not zero.
              label="Refused accesses"
              value={
                metadata.authorization_failures
                  ? String(metadata.authorization_failures)
                  : null
              }
            />
            <Row label="Request" value={metadata.request_id} mono />
          </dl>

          {/*
            Only when something actually returned nothing. A row of green
            ticks on every turn is decoration; the one case worth a reader's
            attention is a tool that ran, succeeded, and matched nothing —
            which is what an unresolved search term looks like, and what the
            `Tools` row above cannot express.
          */}
          {metadata.tool_calls?.some((call) => !call.ok) && (
            <div className={styles.stages}>
              {metadata.tool_calls.map((call, index) => (
                <span
                  key={`${call.name}-${index}`}
                  className={styles.stage}
                  title={call.ok ? "returned rows" : "ran, matched nothing"}
                >
                  {call.ok ? "" : "no match: "}
                  {call.name} <b>{formatMs(call.ms)}</b>
                </span>
              ))}
            </div>
          )}

          {Object.keys(metadata.stage_ms).length > 0 && (
            <div className={styles.stages}>
              {Object.entries(metadata.stage_ms).map(([stage, ms]) => (
                <span key={stage} className={styles.stage}>
                  {stage} <b>{formatMs(ms)}</b>
                </span>
              ))}
            </div>
          )}

          {metadata.generated_sql && (
            <div className={styles.sqlBlock}>
              <p className={styles.sqlLabel}>Generated SQL</p>
              <pre className={styles.sql}>
                <code>{metadata.generated_sql}</code>
              </pre>
            </div>
          )}
        </details>
      )}
    </div>
  );
}

function Row({
  label,
  value,
  mono,
}: {
  label: string;
  value: string | null | undefined;
  mono?: boolean;
}) {
  // An absent value is omitted rather than rendered as "—" or "0". Several
  // of these are legitimately unavailable (token counts on a streamed turn,
  // cost for an unpriced model), and a zero would read as a measurement.
  if (value === null || value === undefined || value === "") return null;
  return (
    <>
      <dt className={styles.key}>{label}</dt>
      <dd className={mono ? styles.valueMono : styles.value}>{value}</dd>
    </>
  );
}

/**
 * The traversal this turn ran, in words, or null if it ran none.
 *
 * Reads the tool name rather than the route: a turn can be labelled KG and
 * still have run no traversal — the graph planner can fail, or the graph can
 * be unreachable — and showing a traversal name for a turn that never
 * executed one would be the panel asserting something that did not happen.
 */
function traversalOf(metadata: ChatMetadata): string | null {
  const call = metadata.tools_used.find((name) => name.startsWith("kg:"));
  if (!call) return null;
  const intent = call.slice("kg:".length);
  return TRAVERSAL_LABEL[intent] ?? intent;
}

function formatMs(ms: number): string {
  return ms >= 1000 ? `${(ms / 1000).toFixed(1)}s` : `${Math.round(ms)}ms`;
}

function formatCost(usd: number | null): string | null {
  // Null means "not priced", which is different from free — see
  // backend/app/llm/pricing.py.
  if (usd === null) return null;
  if (usd === 0) return "$0.00";
  return usd < 0.01 ? `$${usd.toFixed(5)}` : `$${usd.toFixed(3)}`;
}

function formatDate(iso: string | null): string {
  if (!iso) return "";
  const when = new Date(iso);
  if (Number.isNaN(when.getTime())) return "";
  return when.toLocaleDateString(undefined, {
    year: "numeric",
    month: "short",
    day: "numeric",
  });
}
