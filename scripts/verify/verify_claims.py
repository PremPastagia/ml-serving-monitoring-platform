#!/usr/bin/env python
"""Check that the documentation does not claim more than the repository can prove.

Three independent checks, all of which read the actual result files:

1. **Evidence table integrity.** Every row of the evidence table in
   CV_POINTERS_ML_SERVING_MONITORING.md must name a file that exists, a numeric result
   that actually appears in that file, and a reproduction command whose script exists.
2. **No unearned claims.** A small set of phrases ("production-ready", "100%
   reproducible", "zero-downtime", ...) may only appear next to an explicit
   qualification. This is what stops a confident adjective from drifting into the CV.
3. **Deliverables present.** Every document and result file the project promises
   exists and is non-trivial.

Run it after changing any document:

    python scripts/verify/verify_claims.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

CV_FILE = ROOT / "CV_POINTERS_ML_SERVING_MONITORING.md"

#: Phrases that overstate what this project demonstrates. Each maps to the wording
#: that must appear within the same line for the use to be legitimate -- that is, the
#: claim is only allowed when it is being explicitly qualified or denied.
FORBIDDEN_UNLESS_QUALIFIED: dict[str, tuple[str, ...]] = {
    "production-ready": ("not production-ready", "NOT production-ready", "is not a production",
                         "would need", "stops short of", "не"),
    "production ready": ("not production ready", "NOT production ready"),
    "100% reproducible": ("not 100% reproducible", "no claim of 100%"),
    "fully reproducible": ("not fully reproducible", "within one machine",
                           "on one machine", "same machine"),
    "zero-downtime": ("not verified", "no zero-downtime", "cannot claim", "not claimed",
                      "would require", "no failed requests"),
    "zero downtime": ("not verified", "no zero downtime", "cannot claim", "not claimed"),
    "battle-tested": (),
    "enterprise-grade": (),
    "infinitely scalable": (),
    "state-of-the-art": ("not state-of-the-art",),
}

#: Documents that must exist and carry real content.
REQUIRED_DOCS = [
    "README.md",
    "PROJECT_SCOPE.md",
    "DATASET_CARD.md",
    "SYSTEM_DESIGN.md",
    "DATA_VALIDATION.md",
    "TRAINING.md",
    "MLFLOW.md",
    "API.md",
    "LOAD_TESTING.md",
    "CI_CD.md",
    "MONITORING.md",
    "DRIFT_DETECTION.md",
    "RETRAINING.md",
    "ROLLBACK.md",
    "TEST_RESULTS.md",
    "BENCHMARKS.md",
    "FAILURE_ANALYSIS.md",
    "REPRODUCIBILITY.md",
    "FINAL_PROJECT_REPORT.md",
    "INTERVIEW_PREPARATION.md",
    "CV_POINTERS_ML_SERVING_MONITORING.md",
]

REQUIRED_RESULTS = [
    "EVALUATION_RESULTS.csv",
    "SERVING_BENCHMARKS.csv",
    "DRIFT_RESULTS.csv",
    "results/summary.json",
    "results/drift/drift_trials.csv",
    "results/drift/drift_per_feature.csv",
    "results/serving/rollback_through_serving.json",
    "results/retraining/retraining_experiment.csv",
    "results/training/last_training_summary.json",
]

MIN_DOC_BYTES = 400

NUMBER = re.compile(r"-?\d+(?:\.\d+)?")


def fail(problems: list[str], message: str) -> None:
    problems.append(message)


def parse_evidence_rows(text: str) -> list[dict]:
    """Extract the evidence table, identified by its header row."""
    rows: list[dict] = []
    in_table = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("|") and "Evidence File" in stripped and "Safe to Mention" in stripped:
            in_table = True
            continue
        if in_table:
            if not stripped.startswith("|"):
                if stripped:
                    in_table = False
                continue
            cells = [c.strip() for c in stripped.strip("|").split("|")]
            if len(cells) < 6 or set(cells[0]) <= {"-", ":", " "}:
                continue
            rows.append({
                "claim": cells[0],
                "evidence_file": cells[1],
                "test": cells[2],
                "result": cells[3],
                "command": cells[4],
                "safe": cells[5],
            })
    return rows


def strip_markup(value: str) -> str:
    return value.replace("`", "").replace("**", "").strip()


def check_evidence_table(problems: list[str]) -> int:
    if not CV_FILE.exists():
        fail(problems, f"{CV_FILE.name} does not exist")
        return 0
    rows = parse_evidence_rows(CV_FILE.read_text())
    if not rows:
        fail(problems, "no evidence table rows found in " + CV_FILE.name)
        return 0

    for index, row in enumerate(rows, start=1):
        label = f"evidence row {index} ({row['claim'][:48]!r})"
        safe = strip_markup(row["safe"]).lower().startswith("yes")

        if not safe:
            # A "No" row exists to record something the project deliberately does NOT
            # claim. Requiring it to cite an evidence file would be backwards: the whole
            # point is that no evidence exists. What it must do is say so explicitly.
            disclaimer = row["result"].lower()
            if not any(word in disclaimer for word in
                       ("not measured", "not tested", "not claimed", "never executed",
                        "one machine only", "not verified", "not run")):
                fail(problems,
                     f"{label}: marked unsafe but does not state why "
                     f"(got {row['result']!r})")
            continue

        # Every listed evidence file must exist.
        files = [strip_markup(f) for f in re.split(r"[,;]| and ", row["evidence_file"]) if f.strip()]
        contents = ""
        for name in files:
            path = ROOT / name
            if not path.exists():
                fail(problems, f"{label}: evidence file {name!r} does not exist")
                continue
            if path.is_file():
                try:
                    contents += path.read_text(errors="ignore")
                except OSError as exc:
                    fail(problems, f"{label}: cannot read {name!r}: {exc}")

        # Every number quoted as a result must appear in the evidence.
        quoted = NUMBER.findall(strip_markup(row["result"]))
        if safe and not quoted and "n/a" not in row["result"].lower():
            fail(problems, f"{label}: marked safe but quotes no measurable result")
        for number in quoted:
            if number in ("0", "1", "2") and len(quoted) > 1:
                continue  # trivially common tokens; the distinctive figures carry the check
            if number not in contents:
                fail(problems,
                     f"{label}: result {number!r} does not appear in {files!r}")

        # The reproduction command must name a script that exists.
        command = strip_markup(row["command"])
        for token in re.findall(r"(?:scripts|tests)/[\w/\-.]+", command):
            target = ROOT / token.split("::")[0]
            if not target.exists():
                fail(problems, f"{label}: reproduction command references missing {token!r}")
        if not command:
            fail(problems, f"{label}: no reproduction command given")
    return len(rows)


#: Words that, in the same line or the two lines after it, show the phrase is being
#: denied, questioned or discussed rather than asserted.
DENIAL_MARKERS = (
    "not ", "no ", "never", "cannot", "can't", "would need", "would require",
    "unqualified", "do not", "don't", "is a claim about", "avoid", "refus",
    "stops short", "0 failed", "no failed",
)

#: Context window searched for a denial: one line back and two forward. A question
#: heading is answered on the line *below* it, and a sentence introducing the phrases
#: this scan looks for often sits on the line *above*, so a single-line check would flag
#: every honest Q&A entry and this script's own documentation.
DENIAL_LOOKBACK = 1
DENIAL_LOOKAHEAD = 2


def check_no_overclaim(problems: list[str]) -> int:
    checked = 0
    for path in sorted(ROOT.glob("*.md")):
        if path.name == Path(__file__).name:
            continue
        checked += 1
        lines = path.read_text(errors="ignore").splitlines()
        for number, line in enumerate(lines, start=1):
            lowered = line.lower()
            for phrase, allowances in FORBIDDEN_UNLESS_QUALIFIED.items():
                if phrase not in lowered:
                    continue
                if any(allow.lower() in lowered for allow in allowances):
                    continue
                # A question is not a claim.
                if "?" in line:
                    continue
                # A table cell recording the phrase as unsafe is not a claim.
                if line.lstrip().startswith("|") and "**no**" in lowered:
                    continue
                start = max(0, number - 1 - DENIAL_LOOKBACK)
                context = " ".join(lines[start:number + DENIAL_LOOKAHEAD]).lower()
                if any(marker in context for marker in DENIAL_MARKERS):
                    continue
                fail(problems,
                     f"{path.name}:{number}: unqualified claim {phrase!r} -> {line.strip()[:110]}")
    return checked


def check_deliverables(problems: list[str]) -> int:
    present = 0
    for name in REQUIRED_DOCS:
        path = ROOT / name
        if not path.exists():
            fail(problems, f"required document {name} is missing")
        elif path.stat().st_size < MIN_DOC_BYTES:
            fail(problems, f"required document {name} is only {path.stat().st_size} bytes")
        else:
            present += 1
    for name in REQUIRED_RESULTS:
        path = ROOT / name
        if not path.exists():
            fail(problems, f"required result file {name} is missing")
        elif path.stat().st_size == 0:
            fail(problems, f"required result file {name} is empty")
        else:
            present += 1
    return present


def main() -> int:
    problems: list[str] = []
    rows = check_evidence_table(problems)
    docs = check_no_overclaim(problems)
    files = check_deliverables(problems)

    print(f"evidence rows checked : {rows}")
    print(f"markdown files scanned: {docs}")
    print(f"deliverables present  : {files} of {len(REQUIRED_DOCS) + len(REQUIRED_RESULTS)}")

    if problems:
        print(f"\n{len(problems)} problem(s):")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print("\nVERIFY_CLAIMS_OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
