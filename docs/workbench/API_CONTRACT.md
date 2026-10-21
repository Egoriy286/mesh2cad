# Workbench implementation contract

Local FastAPI API in `api/`; Streamlit app and viewer component in `frontend/`. Python 3.12 `.venv-mesh2cad/bin/python`. UI communicates with API by HTTP; business state is durable, not owned by a Streamlit session. API binds 127.0.0.1:8765; Streamlit binds 127.0.0.1:8501. No paid smoke tests or secrets in files.

## JSON entities

- Asset: `id,name,format,created_at,stats,normalization,mesh_url,image_url`.
- Candidate: `id,asset_id,parent_id,label,status,created_at,code,metrics,error,mesh_url,step_url,comparison_url,job_id,source`. Status `valid|execution_error|metric_error`; only finished candidates appear. Immutable content and parent; manual selection is UI state. `metrics` nullable flat dict, see below.
- Job: `id,asset_id,parent_id,kind,status,stage,created_at,updated_at,elapsed_seconds,error,result_candidate_id`. kind `predict|edit|explain|execute|evaluate`; status `queued|running|cancelling|cancelled|complete|failed|interrupted`. Capture selected parent at submit; never mutate it from later selection.
- Message: `id,asset_id,candidate_id,role,content,created_at,job_id,related_candidate_id`. Plain text, never executable HTML.
- Budget: `limit_usd,known_usd,reserved_usd,available_usd,unknown_calls`. Global persisted reservation across explain/edit/generate/retries. Initial prior spent/reserved imported from existing run ledgers when available. Missing usage stays reserved.
- Settings: `provider` (`demo|openrouter`), `model`, `reasoning_effort`, `api_key_configured` boolean, `budget_limit_usd`, `max_attempts` (1–3). Secrets setter only, no returned key. `codex` requests are explicitly rejected; the existing CLI remains its supported entry point. OpenRouter currently allows only `openai/gpt-6-luna` with enforced price bounds.

## Routes

- `GET /health` -> `{status:"ok",...}`.
- `GET /api/state?asset_id=...` -> `{assets,candidates,jobs,messages,budget,settings}`. Without asset ID use most recent asset (or empty arrays). Assets always lists all assets. Candidates/messages filter selected asset.
- `GET /api/examples` -> `{examples:[{id,name,description}]}`.
- `POST /api/examples/{id}` -> Asset; load trusted deterministic geometry and clearly marked local demo candidates. Do not fetch network resources or run model calls.
- `POST /api/assets` multipart `file` -> Asset. Accept STL/OBJ/PLY/OFF/GLB triangular surfaces, reject point-only with clear reason. Keep GLB scene transforms; disallow external OBJ resource resolution.
- `GET /api/assets/{id}/mesh` -> normalized STL; `/image` -> PNG.
- `GET /api/geometry?asset_id=...&candidate_id=...` -> `{frame_id,target:{vertices,faces},candidate:{vertices,faces}|null,preview:{...}}`. Vertex array `[[x,y,z],...]`, face array triangle indices. Rendering uses this data, same coordinates, no independent alignment. Candidate optional. Preview limits never alter evaluation geometry.
- `POST /api/jobs` JSON `{kind,asset_id,parent_id?,prompt?,code?,provider?,model?,reasoning_effort?,max_attempts?,n_points?,seed?,threshold?}` -> Job (HTTP 202). UI generates `Idempotency-Key` header per submit; same key never duplicates a model call/job. If settings unsupported, reject explicitly.
- `POST /api/jobs/{id}/cancel` -> Job. Queued cancel terminal immediately; running acknowledges promptly and terminates stage safely.
- `GET /api/candidates/{id}/mesh|step|code|comparison` -> artifact; reject unsupported/missing artifacts with explicit 404. No arbitrary paths.
- `GET /api/assets/{id}/report?format=json|csv` -> metadata and all evaluations with their parameters; consumers must compare compatible configurations.
- `GET /api/settings` -> settings + budget.
- `POST /api/settings` JSON subset settings + optional `api_key` -> settings + budget, no key in response. Key kept only server memory/environment, never durable state.

Use standard HTTP `detail` error messages, UI displays actionable error. Extra fields permitted when useful; do not break these names without coordinating.

## Evaluation schema

`api.evaluation.evaluate_pair(target_path, prediction_path, n_points=8192, seed=42, threshold=0.01) -> dict` is owned by root integrator. Flat values: `chamfer,iou,iou_status,precision,recall,f_score,h95,volume_relative_error,bbox_extents_error,target_watertight,prediction_watertight,n_points,seed,threshold,metric_version,target_sha256,prediction_sha256`, plus diagnostics/stats if needed. Missing metrics are null with reasons; no fabricated zero. `h95` max of both directional 95th percentiles; volume relative error only if valid volumes. All metrics unaligned in normalized target coordinates.

## Ownership and integration

- Engine builder owns `api/*` except `api/evaluation.py`; backend API/service/jobs/import/storage tests `tests/test_workbench_api.py` (and own additional `test_workbench_*` files except evaluation).
- UI builder owns `frontend/*` and UI browser tests/scripts, no root pyproject or API edits.
- Root owns `api/evaluation.py`, executor cancellation extension, pyproject/lockfile, launch script, README, integration, browser QA and final review.
- Reference/evidence helper writes only `.cache/gauntlet`, `.cache/workbench-qa` and `docs/workbench/REFERENCE_STATUS.md`. No source edits.
- Keep existing CLI and historical results intact. Share interface changes before implementing them. No agent receives API keys.
