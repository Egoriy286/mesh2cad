# Frozen reference evidence — 21 October 2026

The selected references have different roles: Google AI Studio is the organizational bar; Google DeepMind is visual direction. No critic verdict or win has been recorded. The run is active with all gates pending.

## Actual evidence

Local evidence directory: `.cache/workbench-qa/references/`. `manifest.json` records source URLs, SHA-256 hashes, image dimensions, capture conditions and limitations.

- [AI Studio official documentation](https://ai.google.dev/gemini-api/docs/aistudio-build-mode): frozen HTML and 1440×900 docs screenshot. This page currently has no workspace screenshots. The docs screenshot is provenance, **not the UI bar**.
- [Official Google Codelab](https://codelabs.developers.google.com/vibe-code-with-gemini-in-aistudio): frozen HTML and its actual published UI screenshots. The full Build workspace is `official-1b3351663df80961.png` (2316×1336). It shows top project/actions, assistant conversation and bottom composer at left, a large preview/code region at right, and settings access. This is the primary organization evidence.
- `aistudio-workspace-scaled-1440x831.png` is an explicitly scaled derivative of that complete official PNG. It does not establish a native 1440×831 viewport or known original DPR.
- Live `https://aistudio.google.com/apps` redirected to Google Account sign-in in a fresh isolated browser context. `aistudio-live-access-1440x900.png` records that access limit. No credentials or paid APIs were used.
- [Google DeepMind](https://deepmind.google/): frozen HTML and live screenshots `deepmind-1440x900.png`, `deepmind-1280x800.png`, `deepmind-390x844.png`. The screenshots show the public site's actual airy typography, subtle outlined controls and blue visual treatment. It is a marketing page, not a comparable CAD workspace.

## Scope and comparability

The AI Studio screenshot permits a **static application-shell hierarchy comparison**: can a reader distinguish project context, dominant working canvas, assistant/composer, and settings/actions? It does not permit judgment of CAD geometry against game art. Chat may be on either side. Source screenshot is dark, while DeepMind direction is light; equivalent theme, content, state and exact viewport cannot be established. The critic must return `invalid` if these differences materially affect preference. This evidence cannot support a broad claim that Mesh2CAD beats the live AI Studio product, or interactive/mobile superiority.

Use the same 1440×831 frame for a future narrow static packet, alongside mandatory native application captures at 1440×900, 1280×800, 768×1024 and 390×844 for behavior/layout checks. Scrub product identity consistently in both exports before randomization; retain originals separately. Do not pass a branded screenshot to a critic while claiming full blindness. No mapping goes into a ledger or critic prompt. DeepMind style checking is separate from AI Studio organizational comparison.

## Local Gauntlet setup

Pinned upstream: `thors-github/gauntlet-loop-codex` commit `a674e37d58d34e8e50b858fb7bd1b1bb8dc0b0cc`. Only its skill folder was copied to `.cache/gauntlet/`; no global installation, hooks, automation, dependencies or app-source changes. Both helper scripts and critic contract were inspected before execution. The standard-library ledger helper self-test passed all 13 checks.

Run: `.cache/gauntlet/runs/msu-ai-school-b8d9a6c3cd0d/20261021-workbench/`.

- `state.json` is authoritative; `history.jsonl` is empty until a real fresh critic returns a baseline judgment.
- `progress.md` is helper-generated; do not hand-edit it.
- `.cache/workbench-qa/gate-contract.json` lists all 12 mandatory gate names derived from PLAN.md. Pending/absent evidence is not passing. All must pass for `won`; `na` is insufficient.
- `.cache/workbench-qa/critic-prompt-template.md` is the read-only future fresh critic contract. Fill only neutral paths and actual evidence; withhold source identities, rationale, prior verdicts and mapping. A final overall critic is required after piece reviews.

Use `.venv-mesh2cad/bin/python` for helper commands. Read status before resuming. Record actual baseline and subsequent verdicts with all gate results, frozen bar and candidate hashes. Do not fabricate baseline evidence before the app is captured. Lead must preserve a recoverable baseline/checkpoint of current app changes before further loop edits.

```sh
.venv-mesh2cad/bin/python .cache/gauntlet/scripts/gauntlet_state.py status --run .cache/gauntlet/runs/msu-ai-school-b8d9a6c3cd0d/20261021-workbench
.venv-mesh2cad/bin/python .cache/gauntlet/scripts/prepare_blind.py --ours <identity-scrubbed-ours.png> --bar <identity-scrubbed-bar.png> --output-dir <new-neutral-directory>
```

Keep the printed randomized mapping only in lead context. No subjective preference can override failed correctness, budget, safety, persistence or accessibility gates. Ties, invalid comparisons and low confidence are non-wins. `finish --status won` is authorized by the protocol only after a medium/high-confidence overall preference for ours, every gate passing, and unchanged judged artifact/reference hashes.
