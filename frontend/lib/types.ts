// Mirrors backend/app/schemas/chat.py and schemas/clinical.py.
//
// Hand-written rather than generated from the OpenAPI schema: the surface is
// small, and a generator would be one more build step to keep working. If
// this drifts, the symptom is a TypeScript error at the call site, which is
// where the mismatch is easiest to read.

export type Route =
  | "API"
  | "RAG"
  // Relationship and multi-hop questions, answered from the Neo4j
  // projection. Must stay in step with `ROUTES` in
  // backend/app/models/enums.py and the CHECK constraint migration 0009
  // put on messages.route — a route the backend can emit and this union
  // cannot name is a runtime value with no type.
  | "KG"
  | "HYBRID"
  | "TEXT_TO_SQL"
  | "ACTION"
  | "OUT_OF_SCOPE";

export interface Source {
  document_id: number | null;
  encounter_id: number | null;
  chunk_id: number | null;
  document_type: string | null;
  title: string | null;
  section: string | null;
  date: string | null;
  score: number | null;
}

/**
 * One tool invocation and how it went.
 *
 * `ok` is the field that earns this type its place beside `tools_used`. A
 * traversal that ran and matched nothing has the same name as one that
 * returned the whole care team, and the names list cannot tell them apart —
 * which is precisely what a developer panel exists to show.
 */
export interface ToolCall {
  name: string;
  ms: number;
  ok: boolean;
}

export interface ChatMetadata {
  request_id: string;
  route: Route;
  /**
   * The model asked for, and the one the provider reported answering with.
   * PRD §26 names both. They differ when an alias resolves server-side, and
   * `model_version` is null on a streamed turn — which is most of them —
   * because streaming never returns a response object to read it from.
   */
  model: string | null;
  model_version: string | null;
  latency_ms: number;
  stage_ms: Record<string, number>;
  tools_used: string[];
  tool_calls: ToolCall[];
  input_tokens: number | null;
  output_tokens: number | null;
  estimated_cost_usd: number | null;
  retrieved_chunks: number | null;
  reranked_chunks: number | null;
  reranker: string | null;
  deduplicated_chunks: number | null;
  /**
   * Whether the question was rewritten to stand alone before retrieval
   * (PRD §19). The flag only — never either form of the question.
   */
  query_rewritten: boolean;
  generated_sql: string | null;
  sql_row_count: number | null;
  action: string | null;
  /**
   * Schema-constrained model calls on this turn, and how many produced JSON
   * that failed to parse. Both null when the turn made none — a rule-routed
   * question asks the model for no JSON at all, and "0/0 valid" would report
   * a measurement that never happened.
   */
  structured_calls: number | null;
  structured_failures: number | null;
  /**
   * Graph nodes executed, data accesses refused, and checks failed (PRD §26).
   * Null rather than 0 for the same reason as above.
   */
  agent_iterations: number | null;
  authorization_failures: number | null;
  validation_failures: number | null;
  guardrails: string[];
  router_enabled: boolean;
  llm_provider: string | null;
}

/** A write the assistant has proposed but not performed (PRD §11). */
export interface PendingAction {
  action: string;
  /** What the patient is agreeing to, composed by the backend. */
  summary: string;
  /** Opaque and signed. Passed back untouched; never parsed here. */
  token: string;
  expires_at: string;
}

export interface ChatResponse {
  answer: string;
  route: Route;
  conversation_id: string;
  sources: Source[];
  metadata: ChatMetadata;
  pending_action: PendingAction | null;
  disclaimer: string;
}

export interface ActionResult {
  status: "executed" | "declined";
  message: string;
  appointment_id: number | null;
}

export interface DemoAccount {
  email: string;
  password: string;
  display_name: string;
  patient_external_id: string;
}

export interface PatientProfile {
  id: number;
  external_id: string;
  first_name: string;
  last_name: string;
  date_of_birth: string;
}

export interface Appointment {
  id: number;
  appointment_date: string;
  appointment_type: string;
  status: string;
  provider_name: string | null;
  notes: string | null;
}

export interface Medication {
  id: number;
  name: string;
  dosage: string | null;
  frequency: string | null;
  status: string;
  start_date: string | null;
  end_date: string | null;
}

export interface LabResult {
  id: number;
  test_name: string;
  value: number | string;
  unit: string | null;
  reference_range: string | null;
  result_date: string;
}

/** GET /health. Unauthenticated, and the only place the UI learns which
 * model is actually answering — see HealthResponse in the backend. */
export interface Health {
  status: string;
  environment: string;
  database: string;
  vector_backend: string;
  version: string;
  model: string;
  llm_provider: string;
}

export interface Page<T> {
  items: T[];
  count: number;
}

/** One turn as the UI holds it — not a wire type. */
export interface Turn {
  id: string;
  role: "user" | "assistant";
  text: string;
  sources?: Source[];
  metadata?: ChatMetadata;
  pendingAction?: PendingAction | null;
  /** Set once the patient confirms or dismisses, so the card stops asking. */
  actionOutcome?: string;
  streaming?: boolean;
  error?: string;
}
