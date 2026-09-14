/**
 * The banner PRD §1 requires the UI to display, verbatim.
 *
 * The backend sends this same string with every turn (in the `meta` and
 * `done` frames), and that copy is the one shown once a conversation is
 * under way. This constant exists for the moments before the first frame
 * arrives — the sign-in screen, and an empty chat — which is exactly when a
 * first-time reader is deciding what they are looking at. The two must stay
 * identical: a paraphrase here would mean the mandated wording appears only
 * after someone has already asked a question.
 *
 * Backend source: `DEMO_DISCLAIMER` in backend/app/auth/demo.py.
 */
export const DEMO_DISCLAIMER =
  "DEMO — Uses synthetic patient data. Not for medical diagnosis or treatment.";
