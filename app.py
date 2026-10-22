"""Dumb-simple STL viewer and a fully visible LLM data pipeline."""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st
from PIL import Image

from mesh2cad.geometry import (
    VIEW_SIZE,
    load_mesh,
    mesh_stats,
    normalize_mesh,
    render_views_bytes,
)
from mesh2cad.pipeline import (
    DEFAULT_MODEL,
    REASONING_EFFORTS,
    api_key_from_environment,
    build_payload,
    call_openrouter,
    extract_code,
    payload_digest,
    preview_payload,
    save_run,
)
from mesh2cad.prompts import DEFAULT_REQUEST, SYSTEM_PROMPT

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "data" / "deepcad"
RUNS_DIR = ROOT / "runs" / "ui"

st.set_page_config(page_title="STL → LLM", layout="wide", initial_sidebar_state="collapsed")
st.title("STL → LLM")
st.caption(
    "Сверху — только импорт STL и просмотр. Вся работа ниже, в pipeline: "
    "каждый шаг виден, а полный запрос можно скачать целиком."
)


def dataset_options() -> list[tuple[Path, str]]:
    """Return (mesh path, human label) pairs from the local DeepCAD manifest."""
    mesh_dir = DATASET_DIR / "meshes"
    if not mesh_dir.is_dir():
        return []
    labels: dict[str, str] = {}
    manifest_path = DATASET_DIR / "manifest.json"
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            manifest = {}
        for item in manifest.get("items", []):
            if isinstance(item, dict) and item.get("id") and item.get("source_file"):
                labels[item["id"]] = f"{item['source_file']} · {item.get('split', '?')}"
    return [(path, labels.get(path.stem, path.stem)) for path in sorted(mesh_dir.glob("*.stl"))]


def read_source() -> tuple[str, bytes] | None:
    mode = st.radio(
        "Источник STL",
        ["Файл", "Датасет DeepCAD"],
        horizontal=True,
        key="source_mode",
    )
    if mode == "Файл":
        uploaded = st.file_uploader("Выберите STL", type=["stl"])
        if not uploaded:
            return None
        return uploaded.name, uploaded.getvalue()

    options = dataset_options()
    if not options:
        st.warning("В data/deepcad/meshes нет STL. Загрузите файл или добавьте датасет.")
        return None
    by_path = {str(path): label for path, label in options}
    selected = st.selectbox(
        "STL из data/deepcad",
        list(by_path),
        format_func=by_path.get,
        key="dataset_file",
    )
    path = Path(selected)
    return path.name, path.read_bytes()


@st.cache_data(show_spinner=False, max_entries=8)
def prepare(payload: bytes) -> dict:
    source = load_mesh(payload)
    normalized, normalization = normalize_mesh(source)
    image = render_views_bytes(normalized, VIEW_SIZE)
    return {
        "source_vertices": source.vertices,
        "source_faces": source.faces,
        "source_stats": mesh_stats(source),
        "normalized_stats": mesh_stats(normalized),
        "normalization": normalization,
        "image": image,
        "normalized_stl": normalized.export(file_type="stl"),
    }


def viewer(vertices, faces, revision: str):
    figure = go.Figure(
        go.Mesh3d(
            x=vertices[:, 0],
            y=vertices[:, 1],
            z=vertices[:, 2],
            i=faces[:, 0],
            j=faces[:, 1],
            k=faces[:, 2],
            color="#6ea8fe",
            opacity=1.0,
            flatshading=True,
            lighting={"ambient": 0.45, "diffuse": 0.75, "specular": 0.15, "roughness": 0.7},
            lightposition={"x": 100, "y": 200, "z": 100},
            name="STL",
        )
    )
    figure.update_layout(
        height=560,
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
        paper_bgcolor="rgba(0,0,0,0)",
        scene={
            "aspectmode": "data",
            "xaxis_visible": False,
            "yaxis_visible": False,
            "zaxis_visible": False,
            "dragmode": "orbit",
        },
        uirevision=revision,
    )
    return figure


st.session_state.setdefault("model", DEFAULT_MODEL)
st.session_state.setdefault("reasoning_effort", "medium")
st.session_state.setdefault("system_prompt", SYSTEM_PROMPT)
st.session_state.setdefault("request_text", DEFAULT_REQUEST)
st.session_state.setdefault("api_key", api_key_from_environment(ROOT))

source = read_source()
if source is None:
    st.info("Загрузите STL или выберите файл из датасета — появится просмотр и pipeline.")
    st.stop()

source_name, source_bytes = source
source_sha = hashlib.sha256(source_bytes).hexdigest()
data = prepare(source_bytes)
source_stats = data["source_stats"]

view_col, info_col = st.columns([3, 1], gap="large")
with view_col:
    st.plotly_chart(
        viewer(data["source_vertices"], data["source_faces"], source_sha),
        config={"displayModeBar": False, "scrollZoom": True},
        width="stretch",
        key="stl-view",
    )
    st.caption("Вращение — левая кнопка, масштаб — колесо, сдвиг — Shift + левая кнопка.")
with info_col:
    st.metric("Треугольники", f"{source_stats['faces']:,}")
    st.metric("Вершины", f"{source_stats['vertices']:,}")
    st.metric("Watertight", "да" if source_stats["watertight"] else "нет")
    st.caption(f"Файл: {source_name}")
    st.download_button(
        "Скачать исходный STL",
        source_bytes,
        file_name=source_name,
        mime="model/stl",
        width="stretch",
    )

st.divider()
st.header("Pipeline")
st.caption(
    "Ниже нет скрытых преобразований: нормализация, изображение 512×512, текст запроса "
    "и точный JSON для LLM. Ответ модели сохраняется рядом с исходниками."
)

# ---------------------------------------------------------------------------
# 1. Normalization
# ---------------------------------------------------------------------------
st.subheader("1 · Нормализация")
st.caption(
    "Только вход нормализуется: центр bounding box → 0, длиннейшая сторона → 1. "
    "Матрица и границы показаны полностью."
)
norm = data["normalization"]
normalized_stats = data["normalized_stats"]
n1, n2 = st.columns([1, 1], gap="large")
with n1:
    st.markdown("**Матрица source → normalized**")
    st.code(json.dumps(norm["transform"], indent=2), language="json")
with n2:
    st.markdown("**Границы и размеры**")
    st.json(
        {
            "source_bounds": norm["source_bounds"],
            "normalized_bounds": norm["normalized_bounds"],
            "normalized_extents": normalized_stats["extents"],
            "normalized_watertight": normalized_stats["watertight"],
            "normalized_volume": normalized_stats["volume"],
        }
    )
st.download_button(
    "Скачать normalized.stl",
    data["normalized_stl"],
    file_name=f"{Path(source_name).stem}-normalized.stl",
    mime="model/stl",
)

# ---------------------------------------------------------------------------
# 2. Image for the LLM
# ---------------------------------------------------------------------------
st.subheader("2 · Изображение для LLM — 512 × 512")
image_bytes = data["image"]
with Image.open(io.BytesIO(image_bytes)) as image:
    image_size = image.size
image_sha = hashlib.sha256(image_bytes).hexdigest()
i1, i2 = st.columns([1, 1], gap="large")
with i1:
    st.image(image_bytes, width=VIEW_SIZE, caption="llm-input-512.png")
with i2:
    st.markdown(
        f"**Фактический размер:** {image_size[0]} × {image_size[1]} px  \n"
        f"**Формат:** PNG, {len(image_bytes):,} байт  \n"
        "**Раскладка:** 2×2 — front, side, top, isometric."
    )
    st.code(image_sha, language=None)
    st.download_button(
        "Скачать PNG 512×512",
        image_bytes,
        file_name="llm-input-512.png",
        mime="image/png",
    )
    st.caption(
        "Одинаковый кадр [-0.6, 0.6] для всех четырёх ортографических видов. "
        "Подписи видов впечатаны в изображение."
    )

# ---------------------------------------------------------------------------
# 3. Text input and exact request
# ---------------------------------------------------------------------------
st.subheader("3 · Ввод и запрос для LLM")
st.caption(
    "System prompt и запрос пользователя попадают в запрос без переписывания. "
    "Измените их здесь и проверьте точный JSON ниже."
)
c1, c2 = st.columns([1, 2], gap="large")
with c1:
    st.text_input("Модель OpenRouter", key="model")
    st.selectbox(
        "Reasoning effort",
        REASONING_EFFORTS,
        index=REASONING_EFFORTS.index(st.session_state["reasoning_effort"]),
        key="reasoning_effort",
    )
    st.caption("max_tokens = 4096; response_format = json_object.")
with c2:
    st.text_area("System prompt", key="system_prompt", height=260)
st.text_area("Запрос пользователя", key="request_text", height=90)

payload = build_payload(
    model=st.session_state["model"],
    system_prompt=st.session_state["system_prompt"],
    request=st.session_state["request_text"],
    image_png=image_bytes,
    extents=normalized_stats["extents"],
    reasoning_effort=st.session_state["reasoning_effort"],
)
request_sha = payload_digest(payload)
request_json = json.dumps(payload, ensure_ascii=False, indent=2)
st.markdown("**Точный запрос к LLM** (base64 сокращён только в этом предпросмотре)")
st.json(preview_payload(payload))
st.caption(
    f"Полный JSON: {len(request_json.encode('utf-8')):,} байт; "
    f"base64-изображение: {len(payload['messages'][1]['content'][1]['image_url']['url']):,} символов. "
    f"SHA-256 запроса: {request_sha}"
)
st.download_button(
    "Скачать request.json",
    request_json.encode("utf-8"),
    file_name="request.json",
    mime="application/json",
)

# ---------------------------------------------------------------------------
# 4. Model call and raw response
# ---------------------------------------------------------------------------
st.subheader("4 · Ответ LLM")
st.caption(
    "Кнопка отправляет ровно тот JSON, который показан выше, и ничего не добавляет. "
    "Ключ живёт только в памяти этой сессии и не попадает в артефакты."
)
api_key = st.text_input("OpenRouter API key", key="api_key", type="password")
send = st.button(
    "Отправить в OpenRouter",
    type="primary",
    disabled=not api_key.strip(),
)

if send:
    st.session_state.pop("result_error", None)
    with st.spinner("Жду ответ OpenRouter…"):
        try:
            response = call_openrouter(payload, api_key)
            code = extract_code(response)
        except Exception as exc:  # noqa: BLE001 - the message is shown to the user
            st.session_state["result_error"] = str(exc)
        else:
            result = {
                "response": response,
                "code": code,
                "run_dir": None,
                "source_sha": source_sha,
                "request_sha": request_sha,
            }
            try:
                result["run_dir"] = str(
                    save_run(
                        RUNS_DIR,
                        source_name=source_name,
                        source_bytes=source_bytes,
                        image_png=image_bytes,
                        payload=payload,
                        response=response,
                        code=code,
                    )
                )
            except Exception as exc:  # noqa: BLE001 - keep the paid answer visible
                st.session_state["result_error"] = (
                    f"Ответ получен, но артефакты не сохранены: {exc}"
                )
            st.session_state["result"] = result

if st.session_state.get("result_error"):
    st.error(st.session_state["result_error"])

result = st.session_state.get("result")
if result:
    if result["source_sha"] != source_sha:
        st.warning("Ответ относится к другому STL. Загрузите тот же файл или отправьте заново.")
    elif result["request_sha"] != request_sha:
        st.warning("Ответ относится к другому состоянию полей. Отправьте запрос заново.")
    if result["run_dir"]:
        st.success(f"Сохранено: {result['run_dir']}")
    else:
        st.warning("Ответ получен, но файлы запуска не сохранены.")
    st.markdown("**Код CadQuery**")
    st.code(result["code"], language="python")
    st.download_button(
        "Скачать code.py",
        result["code"].encode("utf-8"),
        file_name="code.py",
        mime="text/x-python",
    )
    st.markdown("**Сырой ответ OpenRouter**")
    st.json(result["response"])
    usage = result["response"].get("usage")
    if isinstance(usage, dict):
        st.markdown("**Usage**")
        st.json(usage)

st.divider()
st.caption(
    "Сохранённый запуск лежит в runs/ui/<время>/: source.stl, llm-input-512.png, "
    "request.json, response.json, code.py, meta.json."
)
