"""Clinical content catalogue for the synthetic data generator.

Every string in this file is fabricated. It describes plausible *shapes* of
clinical documentation — a chief complaint followed by a history, an exam, an
assessment and a plan — because retrieval quality is dominated by document
structure, not by medical accuracy. Nothing here is derived from a real
record and nothing here should be read as clinical guidance.

The catalogue is kept separate from :mod:`generate_data` so that the
generator stays readable as a program and the content stays readable as
content. Adding a condition means editing one table, not the generator.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class MedSpec:
    """A medication and the dose ladder a prescriber might move along.

    ``doses`` is ordered from lowest to highest. The generator escalates by
    one step when an encounter changes therapy, which is what makes
    "Metformin increased from 500mg to 1000mg" fall out of the data rather
    than being asserted by a prompt.
    """

    name: str
    doses: tuple[str, ...]
    frequency: str


@dataclass(frozen=True, slots=True)
class LabSpec:
    test_name: str
    unit: str
    reference_range: str
    #: (low, high) for a result inside the reference range.
    normal: tuple[float, float]
    #: (low, high) for an out-of-range result.
    abnormal: tuple[float, float]
    decimals: int = 1


@dataclass(frozen=True, slots=True)
class ProcedureSpec:
    """Something done to the patient at a visit, rather than prescribed.

    ``code`` is a fabricated CPT-shaped identifier. It exists because coded
    procedures are how real records express this, and a graph that holds only
    a display name cannot answer "was this the same procedure?" across two
    visits that worded it differently.
    """

    name: str
    code: str


@dataclass(frozen=True, slots=True)
class AllergySpec:
    substance: str
    reaction: str
    severity: str


@dataclass(frozen=True, slots=True)
class Condition:
    key: str
    display: str
    chief_complaints: tuple[str, ...]
    history: tuple[str, ...]
    examination: tuple[str, ...]
    assessment: tuple[str, ...]
    plan: tuple[str, ...]
    medications: tuple[MedSpec, ...] = ()
    labs: tuple[LabSpec, ...] = ()
    #: Probability that a given encounter for this condition changes therapy.
    change_rate: float = 0.35
    aliases: tuple[str, ...] = field(default=())
    #: A fabricated ICD-10-shaped code. What distinguishes a ``Diagnosis``
    #: from a ``Condition`` in PRD §17: the Condition is the entry on the
    #: problem list, the Diagnosis is the coded assertion made at one visit.
    icd10: str = ""
    #: Procedures a visit for this condition might include. Empty for the
    #: conditions managed entirely by prescription and observation — most of
    #: them, which is why a patient's procedure list should be short.
    procedures: tuple[ProcedureSpec, ...] = ()


HBA1C = LabSpec("HbA1c", "%", "4.0-5.6", (5.0, 5.6), (6.5, 9.4), 1)
FASTING_GLUCOSE = LabSpec("Fasting Glucose", "mg/dL", "70-99", (78, 98), (110, 180), 0)
SYSTOLIC = LabSpec("Systolic Blood Pressure", "mmHg", "90-120", (108, 128), (138, 168), 0)
DIASTOLIC = LabSpec("Diastolic Blood Pressure", "mmHg", "60-80", (68, 80), (86, 102), 0)
LDL = LabSpec("LDL Cholesterol", "mg/dL", "<100", (72, 99), (132, 189), 0)
HDL = LabSpec("HDL Cholesterol", "mg/dL", ">40", (45, 68), (28, 39), 0)
TRIGLYCERIDES = LabSpec("Triglycerides", "mg/dL", "<150", (90, 148), (190, 340), 0)
TSH = LabSpec("TSH", "mIU/L", "0.4-4.0", (0.8, 3.6), (5.2, 11.0), 2)
CREATININE = LabSpec("Creatinine", "mg/dL", "0.6-1.3", (0.7, 1.1), (1.5, 2.2), 2)
EGFR = LabSpec("eGFR", "mL/min/1.73m2", ">60", (72, 108), (38, 58), 0)
HEMOGLOBIN = LabSpec("Hemoglobin", "g/dL", "12.0-17.5", (12.6, 16.4), (9.4, 11.6), 1)
VITAMIN_D = LabSpec("Vitamin D", "ng/mL", "30-100", (34, 72), (12, 26), 0)

#: Drawn for every patient at most encounters, regardless of condition.
ROUTINE_LABS: tuple[LabSpec, ...] = (
    SYSTOLIC,
    DIASTOLIC,
    HEMOGLOBIN,
    CREATININE,
    EGFR,
)


CONDITIONS: tuple[Condition, ...] = (
    Condition(
        key="type_2_diabetes",
        display="Type 2 diabetes mellitus",
        aliases=("diabetes", "blood sugar", "glucose control"),
        chief_complaints=(
            "Routine diabetes follow-up.",
            "Follow-up for blood sugar control.",
            "Reports occasional fatigue in the afternoons.",
        ),
        history=(
            "Patient reports fasting readings at home in the 130s to 150s. "
            "Adherence to the current regimen is described as good, with two "
            "or three missed doses in the past month.",
            "Home glucose log reviewed. Morning values have drifted upward "
            "since the last visit. Diet has been inconsistent during recent "
            "travel; exercise has been limited to occasional walking.",
            "Patient denies polyuria, polydipsia or blurred vision. Reports "
            "some afternoon fatigue but no hypoglycaemic episodes.",
        ),
        examination=(
            "Weight stable since last visit. Feet examined; no ulceration or "
            "callus. Monofilament sensation intact bilaterally.",
            "No acanthosis. Peripheral pulses palpable. Foot inspection "
            "unremarkable.",
        ),
        assessment=(
            "Type 2 diabetes mellitus, suboptimally controlled. Most recent "
            "HbA1c remains above the agreed target.",
            "Type 2 diabetes mellitus with partial response to current "
            "therapy. No evidence of end-organ complication at this visit.",
        ),
        plan=(
            "Increase current oral agent as documented below. Continue home "
            "glucose monitoring twice daily. Repeat HbA1c in three months. "
            "Dietitian referral offered and accepted.",
            "Continue current regimen. Reinforce dietary counselling and "
            "thirty minutes of walking most days. Recheck HbA1c at the next "
            "visit in approximately three months.",
            "Adjust therapy as noted. Advise the patient to report any "
            "symptoms of hypoglycaemia. Annual retinal screening is due and "
            "has been ordered.",
        ),
        medications=(
            MedSpec("Metformin", ("500mg", "1000mg", "1500mg"), "twice daily"),
            MedSpec("Glipizide", ("5mg", "10mg"), "once daily"),
            MedSpec("Sitagliptin", ("50mg", "100mg"), "once daily"),
        ),
        labs=(HBA1C, FASTING_GLUCOSE),
        icd10="E11.9",
        procedures=(
            ProcedureSpec("Diabetic foot examination", "G0245"),
            ProcedureSpec("Diabetic retinal screening", "92250"),
        ),
        change_rate=0.45,
    ),
    Condition(
        key="hypertension",
        display="Essential hypertension",
        aliases=("blood pressure", "hypertension"),
        chief_complaints=(
            "Blood pressure check.",
            "Follow-up for elevated blood pressure.",
            "Reports intermittent headaches in the mornings.",
        ),
        history=(
            "Home readings average in the 140s over 90s. Patient reports "
            "reducing added salt but describes difficulty with this while "
            "eating out.",
            "Blood pressure diary reviewed. Readings are higher on working "
            "days than at weekends. No chest pain, palpitations or "
            "shortness of breath.",
            "Patient reports adherence to the current antihypertensive with "
            "no side effects. Occasional morning headache, self-resolving.",
        ),
        examination=(
            "Seated blood pressure repeated after five minutes of rest. "
            "Heart sounds normal, no murmur. No peripheral oedema.",
            "Cardiovascular examination unremarkable. No carotid bruit. "
            "Fundoscopy deferred to optometry.",
        ),
        assessment=(
            "Essential hypertension, above target on current therapy.",
            "Essential hypertension with reasonable control; readings remain "
            "marginally above the agreed threshold.",
        ),
        plan=(
            "Titrate antihypertensive therapy as documented. Continue home "
            "monitoring and bring the log to the next visit. Repeat renal "
            "function in four weeks.",
            "Continue current dose. Reinforce salt reduction and regular "
            "aerobic activity. Review in three months with a home reading "
            "log.",
        ),
        medications=(
            MedSpec("Lisinopril", ("10mg", "20mg", "40mg"), "once daily"),
            MedSpec("Amlodipine", ("5mg", "10mg"), "once daily"),
            MedSpec("Hydrochlorothiazide", ("12.5mg", "25mg"), "once daily"),
        ),
        labs=(SYSTOLIC, DIASTOLIC),
        icd10="I10",
        procedures=(
            ProcedureSpec("Ambulatory blood pressure monitoring", "93784"),
            ProcedureSpec("Electrocardiogram, 12-lead", "93000"),
        ),
        change_rate=0.4,
    ),
    Condition(
        key="hyperlipidemia",
        display="Hyperlipidaemia",
        aliases=("cholesterol", "lipids"),
        chief_complaints=(
            "Lipid panel review.",
            "Follow-up for raised cholesterol.",
        ),
        history=(
            "Patient reports tolerating statin therapy without myalgia. "
            "Dietary changes have been partial.",
            "No muscle aches or dark urine. Family history of early "
            "cardiovascular disease discussed again.",
        ),
        examination=(
            "No xanthelasma. Cardiovascular examination normal.",
            "General examination unremarkable. Weight unchanged.",
        ),
        assessment=(
            "Hyperlipidaemia, LDL above target on current therapy.",
            "Hyperlipidaemia responding to treatment; LDL improved but not "
            "yet at goal.",
        ),
        plan=(
            "Continue statin at the documented dose. Repeat lipid panel in "
            "twelve weeks. Reinforce dietary modification.",
            "Increase statin dose as documented. Check liver enzymes with "
            "the next lipid panel.",
        ),
        medications=(
            MedSpec("Atorvastatin", ("10mg", "20mg", "40mg"), "once daily at night"),
            MedSpec("Rosuvastatin", ("5mg", "10mg", "20mg"), "once daily"),
        ),
        labs=(LDL, HDL, TRIGLYCERIDES),
        icd10="E78.5",
        change_rate=0.3,
    ),
    Condition(
        key="knee_osteoarthritis",
        display="Osteoarthritis of the knee",
        aliases=("knee pain", "knee", "joint pain"),
        chief_complaints=(
            "Persistent right knee pain.",
            "Left knee pain, worse on stairs.",
            "Knee pain limiting walking distance.",
        ),
        history=(
            "Patient describes knee pain that is worse when climbing stairs "
            "and after prolonged standing. Pain is rated four to six out of "
            "ten and eases with rest. No locking or giving way. No history "
            "of recent injury.",
            "Knee discomfort has increased over the past six weeks, "
            "particularly on descending stairs. Morning stiffness lasts "
            "around fifteen minutes. Over-the-counter analgesia gives "
            "partial relief.",
            "Ongoing knee pain, unchanged since the last review. The patient "
            "reports difficulty with longer walks and has reduced usual "
            "activity as a result.",
        ),
        examination=(
            "Mild effusion of the affected knee. Crepitus on passive "
            "movement. Range of motion preserved but painful at end range. "
            "Ligaments stable. No erythema or warmth.",
            "Tenderness over the medial joint line. No effusion today. Gait "
            "mildly antalgic. Quadriceps bulk reduced on the affected side.",
        ),
        assessment=(
            "Osteoarthritis of the knee with mechanical pain. No features "
            "suggesting inflammatory arthropathy or internal derangement.",
            "Degenerative knee pain, stable. Symptoms remain activity "
            "related.",
        ),
        plan=(
            "Physical therapy referral placed, with a focus on quadriceps "
            "strengthening. Continue simple analgesia as needed. Follow-up "
            "in approximately six weeks; sooner if the knee locks, gives way "
            "or swells acutely.",
            "Continue home exercise programme. Weight management discussed. "
            "Consider imaging if symptoms progress. Review in three months.",
            "Topical anti-inflammatory trialled. Advised on activity pacing. "
            "Orthopaedic opinion to be considered if there is no improvement "
            "after a course of physical therapy.",
        ),
        medications=(
            MedSpec(
                "Ibuprofen", ("200mg", "400mg", "600mg"), "three times daily as needed"
            ),
            MedSpec("Acetaminophen", ("500mg", "1000mg"), "up to four times daily"),
            MedSpec(
                "Diclofenac gel", ("1%", "2%"), "applied topically three times daily"
            ),
        ),
        labs=(),
        icd10="M17.9",
        procedures=(
            ProcedureSpec("Intra-articular knee injection", "20610"),
            ProcedureSpec("Knee radiograph, three views", "73562"),
        ),
        change_rate=0.35,
    ),
    Condition(
        key="hypothyroidism",
        display="Hypothyroidism",
        aliases=("thyroid", "tsh"),
        chief_complaints=(
            "Thyroid function review.",
            "Fatigue and cold intolerance.",
        ),
        history=(
            "Patient reports persistent tiredness and cold intolerance. "
            "Medication is taken fasting, as advised.",
            "Energy levels have improved since the last dose adjustment. No "
            "palpitations or tremor.",
        ),
        examination=(
            "No goitre palpable. Pulse regular at seventy-two beats per "
            "minute. Skin and reflexes unremarkable.",
            "Thyroid not enlarged. No signs of over-replacement.",
        ),
        assessment=(
            "Primary hypothyroidism, currently under-replaced.",
            "Primary hypothyroidism, biochemically euthyroid on the current "
            "dose.",
        ),
        plan=(
            "Adjust levothyroxine as documented and repeat thyroid function "
            "in six weeks.",
            "Continue the current dose. Repeat thyroid function in six "
            "months.",
        ),
        medications=(
            MedSpec("Levothyroxine", ("50mcg", "75mcg", "100mcg"), "once daily"),
        ),
        labs=(TSH, VITAMIN_D),
        icd10="E03.9",
        change_rate=0.35,
    ),
    Condition(
        key="asthma",
        display="Asthma",
        aliases=("asthma", "wheeze", "breathing"),
        chief_complaints=(
            "Review of asthma control.",
            "Increased use of reliever inhaler.",
        ),
        history=(
            "Reliever inhaler used three to four times per week, mainly "
            "after exertion. No nocturnal waking. No recent exacerbation or "
            "course of oral steroids.",
            "Symptoms are worse during the pollen season. Inhaler technique "
            "reviewed and found to be adequate.",
        ),
        examination=(
            "Chest clear on auscultation. No wheeze at rest. Peak flow "
            "recorded and within the patient's usual range.",
            "Respiratory examination normal. No accessory muscle use.",
        ),
        assessment=(
            "Asthma, partially controlled on current therapy.",
            "Asthma, well controlled. No features of exacerbation.",
        ),
        plan=(
            "Step up preventer therapy as documented. Reinforce inhaler "
            "technique and provide an updated action plan. Review in eight "
            "weeks.",
            "Continue current inhalers. Annual review scheduled. Advise "
            "influenza vaccination in season.",
        ),
        medications=(
            MedSpec("Albuterol inhaler", ("90mcg", "180mcg"), "as needed"),
            MedSpec("Fluticasone inhaler", ("50mcg", "100mcg", "250mcg"), "twice daily"),
        ),
        labs=(),
        icd10="J45.909",
        procedures=(
            ProcedureSpec("Spirometry with bronchodilator", "94060"),
        ),
        change_rate=0.3,
    ),
    Condition(
        key="gerd",
        display="Gastro-oesophageal reflux disease",
        aliases=("reflux", "heartburn", "indigestion"),
        chief_complaints=(
            "Heartburn after evening meals.",
            "Review of reflux symptoms.",
        ),
        history=(
            "Burning discomfort behind the sternum, worse when lying flat. "
            "No dysphagia, weight loss or vomiting.",
            "Symptoms improved with the current acid suppression but return "
            "if doses are missed.",
        ),
        examination=(
            "Abdomen soft and non-tender. No mass or organomegaly.",
            "Abdominal examination unremarkable.",
        ),
        assessment=(
            "Gastro-oesophageal reflux disease, symptomatic.",
            "Gastro-oesophageal reflux disease, controlled on therapy. No "
            "alarm features.",
        ),
        plan=(
            "Continue acid suppression at the documented dose. Advise "
            "against late meals and elevate the head of the bed. Review in "
            "eight weeks.",
            "Trial of dose reduction with a view to stopping if symptoms "
            "remain settled.",
        ),
        medications=(
            MedSpec("Omeprazole", ("20mg", "40mg"), "once daily before food"),
            MedSpec("Famotidine", ("20mg", "40mg"), "twice daily"),
        ),
        labs=(),
        icd10="K21.9",
        procedures=(
            ProcedureSpec("Upper gastrointestinal endoscopy", "43235"),
        ),
        change_rate=0.3,
    ),
)

CONDITIONS_BY_KEY = {condition.key: condition for condition in CONDITIONS}

#: Allergies, which belong to the patient rather than to any condition.
#:
#: Unlike everything else in this file these are not tied to a diagnosis —
#: which is the point of modelling them separately. "Am I allergic to
#: anything?" is answered from the patient's own list, and the clinically
#: interesting version, "does anything I take interact with an allergy?",
#: needs the allergy and the medication to be separate nodes that a traversal
#: can meet at the patient.
ALLERGIES: tuple[AllergySpec, ...] = (
    AllergySpec("Penicillin", "Urticarial rash", "moderate"),
    AllergySpec("Sulfa drugs", "Rash and pruritus", "mild"),
    AllergySpec("Ibuprofen", "Gastric upset", "mild"),
    AllergySpec("Codeine", "Nausea and vomiting", "moderate"),
    AllergySpec("Latex", "Contact dermatitis", "mild"),
    AllergySpec("Shellfish", "Lip and tongue swelling", "severe"),
    AllergySpec("Peanuts", "Anaphylaxis", "severe"),
    AllergySpec("Amoxicillin", "Maculopapular rash", "moderate"),
)

#: Severities, ordered, so the generator can weight toward the mild end and a
#: display can sort by seriousness rather than alphabetically.
ALLERGY_SEVERITIES: tuple[str, ...] = ("mild", "moderate", "severe")


def _condition_key_by_medication() -> dict[str, str]:
    """Map each catalogue drug to the one condition it treats.

    The source of the ``TREATS`` edge in the knowledge graph. Resolving it
    here, from the catalogue that already knows, beats threading a condition
    through every medication call site in the generator — and it cannot drift
    from the data, because it *is* the data.

    Raises on a drug that appears under two conditions rather than silently
    keeping one. That ambiguity has no correct answer at this layer: the
    edge would have to come from the prescribing site instead, and failing
    the seed says so immediately rather than shipping a plausible graph with
    a wrong edge in it.
    """
    owner: dict[str, str] = {}
    for condition in CONDITIONS:
        for medication in condition.medications:
            existing = owner.get(medication.name)
            if existing is not None and existing != condition.key:
                raise ValueError(
                    f"{medication.name!r} is listed under both {existing!r} and "
                    f"{condition.key!r}; TREATS can no longer be resolved by name."
                )
            owner[medication.name] = condition.key
    return owner


#: Drug name -> condition key. Short courses are absent by design: an
#: antibiotic course treats an infection the record does not catalogue, so
#: its TREATS edge is genuinely unknown rather than missing.
CONDITION_KEY_BY_MEDICATION = _condition_key_by_medication()

#: Short, finished courses of treatment unrelated to a chronic condition.
#:
#: Real medication histories are mostly *past* prescriptions, and without
#: them every row in the table is either active or a dose that was replaced.
#: These are the only source of ``status='completed'`` in the dataset, so
#: they are also what stops "what am I currently taking?" from being
#: answerable by selecting the whole table.
SHORT_COURSES: tuple[MedSpec, ...] = (
    MedSpec("Amoxicillin", ("500mg",), "three times daily for 7 days"),
    MedSpec("Cephalexin", ("500mg",), "four times daily for 7 days"),
    MedSpec("Azithromycin", ("250mg",), "once daily for 5 days"),
    MedSpec("Ciprofloxacin", ("500mg",), "twice daily for 7 days"),
    MedSpec("Prednisone", ("20mg",), "once daily for 5 days"),
    MedSpec("Naproxen", ("250mg", "500mg"), "twice daily as needed"),
    MedSpec("Ondansetron", ("4mg",), "as needed"),
    MedSpec("Doxycycline", ("100mg",), "twice daily for 10 days"),
)

PROVIDERS: tuple[tuple[str, str], ...] = (
    ("Dr. Sarah Smith", "Internal Medicine"),
    ("Dr. Daniel Okafor", "Family Medicine"),
    ("Dr. Priya Raman", "Endocrinology"),
    ("Dr. Miguel Alvarez", "Cardiology"),
    ("Dr. Hannah Lindqvist", "Rheumatology"),
    ("Dr. Tomas Novak", "Orthopaedics"),
    ("Dr. Aisha Bello", "Respiratory Medicine"),
    ("Dr. Elena Rossi", "Gastroenterology"),
    ("Dr. James Whitfield", "Internal Medicine"),
    ("Dr. Mei-Ling Chen", "Family Medicine"),
)

APPOINTMENT_NOTE_TEMPLATES: tuple[str, ...] = (
    "Routine follow-up.",
    "Booked by the patient online.",
    "Rescheduled from an earlier slot.",
    "Requested after a telephone triage call.",
    "",
)
