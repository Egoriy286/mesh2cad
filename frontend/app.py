"""Streamlit host for the local Mesh2CAD workspace.

The component owns presentation only. All domain actions use the durable HTTP API.
"""

from __future__ import annotations

import base64
import os
from pathlib import Path

import requests
import streamlit as st
import streamlit.components.v1 as components
from streamlit.errors import StreamlitInvalidLayoutContextError

st.set_page_config(
    page_title="Mesh2CAD · Workbench",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="collapsed",
)
st.html("""<style>
[data-testid="stHeader"]{display:none} .block-container{padding:0!important;max-width:none!important}
[data-testid="stAppViewContainer"]{background:#101114} footer{display:none}
iframe[title*="mesh2cad"]{display:block;border:0} [data-testid="stElementContainer"]{margin:0}
</style>""")
API = os.environ.get("MESH2CAD_API_URL", "http://127.0.0.1:8765").rstrip("/")
workspace = components.declare_component(
    "mesh2cad_workspace", path=str(Path(__file__).parent / "component")
)


def request(method: str, path: str, **kwargs):
    try:
        response = requests.request(method, API + path, timeout=20, **kwargs)
        if not response.ok:
            try:
                detail = response.json().get("detail", response.text)
            except ValueError:
                detail = response.text
            raise RuntimeError(f"{response.status_code}: {detail}")
        return response.json()
    except requests.ConnectionError as exc:
        raise RuntimeError(
            "API недоступен на 127.0.0.1:8765. Запустите локальный сервис и повторите действие."
        ) from exc


for key, value in {
    "asset_id": st.query_params.get("asset", ""),
    "candidate_id": st.query_params.get("candidate", ""),
    "draft": "",
    "notice": "",
    "seen_event": "",
    "event_ack": None,
    "geometry_key": None,
    "geometry_data": None,
    "editor": "",
}.items():
    if key not in st.session_state:
        st.session_state[key] = value


@st.fragment(run_every=2)
def screen():
    ss = st.session_state
    try:
        state = request(
            "GET", "/api/state", params={"asset_id": ss.asset_id} if ss.asset_id else {}
        )
        assets = state.get("assets", [])
        if not ss.asset_id and assets:
            ss.asset_id = assets[-1]["id"]
        candidates = state.get("candidates", [])
        # Follow the newest finished candidate only until the user makes a selection.
        if not ss.candidate_id and candidates:
            ss.candidate_id = candidates[-1]["id"]
        if ss.candidate_id and not any(c["id"] == ss.candidate_id for c in candidates):
            ss.candidate_id = ""
        geometry_key = (ss.asset_id, ss.candidate_id)
        if ss.asset_id and geometry_key != ss.geometry_key:
            ss.geometry_data = request(
                "GET",
                "/api/geometry",
                params={
                    "asset_id": ss.asset_id,
                    **({"candidate_id": ss.candidate_id} if ss.candidate_id else {}),
                },
            )
            ss.geometry_key = geometry_key
        if "examples" not in ss:
            ss.examples = request("GET", "/api/examples").get("examples", [])
        state["examples"] = ss.examples
        state["connection"] = "ok"
    except Exception as exc:
        state = ss.get(
            "last_state",
            {
                "assets": [],
                "candidates": [],
                "jobs": [],
                "messages": [],
                "settings": {},
                "budget": {},
            },
        ).copy()
        state["connection"] = "error"
        state["connection_error"] = str(exc)
    if state.get("connection") == "ok":
        ss.last_state = state.copy()
    event = workspace(
        state=state,
        geometry=ss.geometry_data,
        selected_asset=ss.asset_id,
        selected_candidate=ss.candidate_id,
        draft=ss.draft,
        notice=ss.notice,
        event_ack=ss.event_ack,
        api_url=API,
        key="workspace",
        default=None,
    )
    if not event or event.get("id") == ss.seen_event:
        return
    ss.seen_event = event["id"]
    action = event.get("action")
    data = event.get("data", {})
    ss.notice = ""
    success = True
    try:
        if action == "select_asset":
            ss.asset_id, ss.candidate_id = data["id"], ""
            st.query_params["asset"] = ss.asset_id
            st.query_params.pop("candidate", None)
        elif action == "select_candidate":
            ss.candidate_id = data["id"]
            st.query_params["candidate"] = ss.candidate_id
        elif action == "draft":
            if data.get("asset_id", ss.asset_id) == ss.asset_id:
                ss.draft = data.get("text", "")
        elif action == "example":
            examples = ss.get("examples", [])
            example_id = data.get("id") or (examples[0]["id"] if examples else "")
            if not example_id:
                raise RuntimeError("Нет доступных примеров. Проверьте локальный API.")
            asset = request("POST", "/api/examples/" + example_id)
            ss.asset_id, ss.candidate_id = asset["id"], ""
            st.query_params["asset"] = ss.asset_id
            ss.notice = "Локальный пример загружен. Модель не вызывалась."
        elif action == "examples":
            ss.examples = request("GET", "/api/examples").get("examples", [])
        elif action == "upload":
            payload = base64.b64decode(data["content"])
            asset = request(
                "POST",
                "/api/assets",
                files={"file": (data["name"], payload, "application/octet-stream")},
            )
            ss.asset_id, ss.candidate_id = asset["id"], ""
            st.query_params["asset"] = ss.asset_id
            ss.notice = "Геометрия импортирована и нормализована."
        elif action == "job":
            body = dict(data)
            body.setdefault("asset_id", ss.asset_id)
            body.setdefault("parent_id", ss.candidate_id or None)
            request("POST", "/api/jobs", json=body, headers={"Idempotency-Key": event["id"]})
            if (
                body.get("kind") in {"edit", "explain"}
                and body["asset_id"] == ss.asset_id
                and ss.draft.strip() == body.get("prompt", "")
            ):
                ss.draft = ""
            ss.notice = "Задача в очереди. Сцена остаётся доступной."
        elif action == "cancel":
            request("POST", f"/api/jobs/{data['id']}/cancel")
            ss.notice = "Запрос на отмену принят. Ожидаем безопасную остановку."
        elif action == "settings":
            request("POST", "/api/settings", json=data)
            ss.notice = "Настройки сохранены."
        elif action == "check_key":
            if data.get("api_key"):
                request("POST", "/api/settings", json={"api_key": data["api_key"]})
            result = request("POST", "/api/auth/check")
            ss.notice = result["message"]
        elif action == "dismiss":
            ss.notice = ""
    except Exception as exc:
        success = False
        ss.notice = "Ошибка: " + str(exc)
    ss.event_ack = {"id": event["id"], "success": success}
    try:
        st.rerun(scope="fragment")
    except StreamlitInvalidLayoutContextError:
        # A component event can also be delivered during the initial full app
        # run. Streamlit permits fragment-only reruns only inside a fragment run.
        st.rerun()


screen()
