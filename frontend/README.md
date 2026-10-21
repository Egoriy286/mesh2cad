# Mesh2CAD Streamlit frontend

Run `streamlit run frontend/app.py` with the repository virtual environment. The API defaults to `http://127.0.0.1:8765`; override with `MESH2CAD_API_URL`.

The declared Streamlit component keeps a stable iframe. It owns the workspace layout and Three.js scene, and sends unique action IDs to the Python HTTP adapter. The adapter deduplicates component events and uses action IDs as API idempotency keys. A Streamlit fragment polls API state every two seconds without executing work in the UI. No domain engine is duplicated.

Camera position is the only value persisted in browser localStorage. Unsent prompt drafts are stored per asset in browser sessionStorage and restored after refresh or asset changes. Only prompt text and its revision are stored there; keys and settings are excluded. A matching successful server acknowledgement clears the accepted draft without overwriting newer text. API rejection retains the draft. Secrets go directly through the host to the API and are never included in render arguments, URLs, or browser storage. The key input is cleared on submission.

Three.js 0.170.0 and OrbitControls are vendored under `component/vendor/` under the accompanying MIT license. No network connection is required to render the 3D workspace. Source: https://github.com/mrdoob/three.js/tree/r170.

The scene takes priority over the scrollable metrics/history. «Развернуть» temporarily hides the surrounding panels without recreating the canvas or resetting its camera; «Свернуть» or Escape restores them.

Settings offer «Проверить и применить ключ» to validate OpenRouter authorization without inference. Keys stay in API process memory. A confirmed HTTP 401 releases its local call reservation and blocks further paid submissions with the same rejected key; uncertain network/provider failures remain reserved.
