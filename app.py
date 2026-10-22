"""Dumb-simple STL viewer and a fully visible LLM data pipeline."""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st
from PIL import Image

from mesh2cad.executor import execute_code
from mesh2cad.geometry import (
    VIEW_NAMES,
    VIEW_SIZE,
    compose_views,
    evaluate_iou,
    load_mesh,
    mesh_stats,
    normalize_mesh,
    render_llm_images,
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
CADQUERY_AVAILABLE = importlib.util.find_spec("cadquery") is not None

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
    images = render_llm_images(normalized, VIEW_SIZE)
    return {
        "source_vertices": source.vertices,
        "source_faces": source.faces,
        "normalized_vertices": normalized.vertices,
        "normalized_faces": normalized.faces,
        "source_stats": mesh_stats(source),
        "normalized_stats": mesh_stats(normalized),
        "normalization": normalization,
        "images": images,
        "sheet": compose_views(images, VIEW_SIZE),
        "normalized_stl": normalized.export(file_type="stl"),
    }


def _mesh_trace(vertices, faces, *, color: str, opacity: float, name: str):
    return go.Mesh3d(
        x=vertices[:, 0],
        y=vertices[:, 1],
        z=vertices[:, 2],
        i=faces[:, 0],
        j=faces[:, 1],
        k=faces[:, 2],
        color=color,
        opacity=opacity,
        flatshading=True,
        lighting={"ambient": 0.45, "diffuse": 0.75, "specular": 0.15, "roughness": 0.7},
        lightposition={"x": 100, "y": 200, "z": 100},
        name=name,
    )


def viewer(vertices, faces, revision: str, target: tuple | None = None):
    traces = []
    if target is not None:
        traces.append(
            _mesh_trace(
                target[0],
                target[1],
                color="#aab2c0",
                opacity=0.28,
                name="Исходник",
            )
        )
    traces.append(
        _mesh_trace(
            vertices,
            faces,
            color="#6ea8fe",
            opacity=1.0,
            name="CadQuery" if target is not None else "STL",
        )
    )
    figure = go.Figure(data=traces)
    figure.update_layout(
        height=560,
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
        paper_bgcolor="rgba(0,0,0,0)",
        showlegend=target is not None,
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


def _format_volume(value) -> str:
    return "—" if value is None else f"{value:.6g}"


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
# 2. Images for the LLM: three views, RGB + depth
# ---------------------------------------------------------------------------
st.subheader(f"2 · Изображение для LLM — один sheet {VIEW_SIZE * 3}×{VIEW_SIZE * 2}")
images = data["images"]
image_meta: dict[str, dict] = {}
for label, blob in images.items():
    with Image.open(io.BytesIO(blob)) as image:
        image_size = image.size
    if image_size != (VIEW_SIZE, VIEW_SIZE):
        raise RuntimeError(f"{label}: {image_size}, expected {VIEW_SIZE}x{VIEW_SIZE}")
    image_meta[label] = {
        "size": list(image_size),
        "bytes": len(blob),
        "sha256": hashlib.sha256(blob).hexdigest(),
    }
st.caption(
    "Шесть видов: top, left, right, isometric, bottom, front. "
    "В каждом тайле RGB и его depth в одном изображении."
)
sheet = data["sheet"]
with Image.open(io.BytesIO(sheet)) as image:
    sheet_size = image.size
image_meta["llm-input"] = {
    "size": list(sheet_size),
    "bytes": len(sheet),
    "sha256": hashlib.sha256(sheet).hexdigest(),
}
st.image(
    sheet,
    width="stretch",
    caption=f"llm-input · {sheet_size[0]}×{sheet_size[1]}",
)
st.caption(
    "Одно изображение 3×2: top, left, right, isometric, bottom, front. "
    "В каждом тайле RGB и его depth смешаны 50/50 в одно изображение; "
    f"тайл {VIEW_SIZE}×{VIEW_SIZE}, фон чёрный; в depth ближе — светлее, "
    "дальше — темнее; чистый чёрный — пустота."
)
download_col, archive_col = st.columns(2)
with download_col:
    st.download_button(
        "Скачать llm-input.png",
        sheet,
        file_name="llm-input.png",
        mime="image/png",
        width="stretch",
    )
archive_buffer = io.BytesIO()
with zipfile.ZipFile(archive_buffer, "w", zipfile.ZIP_DEFLATED) as archive:
    for label, blob in images.items():
        archive.writestr(f"{label}.png", blob)
    archive.writestr("llm-input.png", sheet)
with archive_col:
    st.download_button(
        "Скачать 12 тайлов + llm-input.png",
        archive_buffer.getvalue(),
        file_name="llm-images-512.zip",
        mime="application/zip",
        width="stretch",
    )
with st.expander("Отдельные тайлы и метаданные"):
    tile_columns = st.columns(3)
    for index, view in enumerate(VIEW_NAMES):
        with tile_columns[index % 3]:
            st.markdown(f"**{view}**")
            st.image(
                images[f"{view}-rgb"],
                width="stretch",
                caption=f"{view} RGB · {VIEW_SIZE}×{VIEW_SIZE}",
            )
            st.image(
                images[f"{view}-depth"],
                width="stretch",
                caption=f"{view} depth · {VIEW_SIZE}×{VIEW_SIZE}",
            )
    st.json(image_meta)

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
    images=[("views-rgb-depth", sheet)],
    extents=normalized_stats["extents"],
    reasoning_effort=st.session_state["reasoning_effort"],
)
request_sha = payload_digest(payload)
request_json = json.dumps(payload, ensure_ascii=False, indent=2)
image_chars = sum(
    len(part["image_url"]["url"])
    for message in payload["messages"]
    if isinstance(message["content"], list)
    for part in message["content"]
    if isinstance(part, dict) and part.get("type") == "image_url"
)
st.markdown("**Точный запрос к LLM** (base64 сокращён только в этом предпросмотре)")
st.json(preview_payload(payload))
st.caption(
    f"Полный JSON: {len(request_json.encode('utf-8')):,} байт; "
    f"одно изображение {sheet_size[0]}×{sheet_size[1]} в base64: {image_chars:,} символов. "
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
                        images={"views-rgb-depth": sheet},
                        extra_images=images,
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

    # -----------------------------------------------------------------------
    # 5. Sandboxed CadQuery execution, visualization, and volume IoU
    # -----------------------------------------------------------------------
    st.subheader("5 · CadQuery: просмотр и IoU")
    st.caption(
        "Код выполняется отдельным процессом. macOS — Seatbelt, если доступен; Windows/Linux — "
        "restricted sandbox с Python audit hooks: сеть, процессы и запись вне папки "
        "запуска блокируются. IoU считается по объёмам, без выравнивания."
    )
    stale = result["source_sha"] != source_sha or result["request_sha"] != request_sha
    if stale:
        st.warning("Код относится к другому STL или запросу. Отправьте запрос заново.")
    if not CADQUERY_AVAILABLE:
        st.warning("CadQuery не установлен. Выполните `uv sync --extra dev`.")
    if st.button(
        "Выполнить в CadQuery и посчитать IoU",
        type="primary",
        disabled=stale or not CADQUERY_AVAILABLE,
    ):
        st.session_state.pop("cad", None)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        if result["run_dir"]:
            cad_dir = Path(result["run_dir"]) / "cadquery"
        else:
            cad_dir = RUNS_DIR / f"{stamp}-cadquery"
        with st.spinner("CadQuery в песочнице…"):
            execution = execute_code(result["code"], cad_dir, timeout=60)
            try:
                (cad_dir / "execution.json").write_text(
                    json.dumps(execution, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8",
                )
            except OSError:
                pass
            cad_state = {
                "status": execution["status"],
                "error": execution.get("error"),
                "execution": execution,
                "run_dir": str(cad_dir),
                "source_sha": result["source_sha"],
                "request_sha": result["request_sha"],
            }
            if execution["status"] == "valid":
                try:
                    target = load_mesh(data["normalized_stl"])
                    candidate = load_mesh(execution["mesh_path"])
                    cad_state.update(
                        candidate_vertices=candidate.vertices,
                        candidate_faces=candidate.faces,
                        target_vertices=data["normalized_vertices"],
                        target_faces=data["normalized_faces"],
                    )
                    iou = evaluate_iou(target, candidate)
                    cad_state["iou"] = iou
                    (cad_dir / "iou.json").write_text(
                        json.dumps(iou, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                        encoding="utf-8",
                    )
                except Exception as exc:  # noqa: BLE001 - show the failure
                    cad_state["metric_error"] = f"{type(exc).__name__}: {exc}"
            st.session_state["cad"] = cad_state

    cad = st.session_state.get("cad")
    if cad and (cad.get("source_sha") != source_sha or cad.get("request_sha") != request_sha):
        st.session_state.pop("cad", None)
        cad = None
    if cad:
        if cad["status"] != "valid":
            st.error(f"CadQuery: {cad['status']}")
            if cad.get("error"):
                st.code(cad["error"], language=None)
        else:
            st.success(
                f"Выполнено за {cad['execution']['elapsed_seconds']:.2f} с · "
                f"изоляция: {cad['execution'].get('sandbox', 'restricted')}"
            )
            iou = cad.get("iou")
            if iou:
                iou_col, target_col, cad_col = st.columns(3)
                value = iou.get("iou")
                iou_col.metric("IoU", "—" if value is None else f"{value:.4f}")
                target_col.metric("Объём исходника", _format_volume(iou.get("target_volume")))
                cad_col.metric("Объём CadQuery", _format_volume(iou.get("prediction_volume")))
                st.caption(
                    f"Статус: `{iou.get('status')}` · "
                    f"пересечение: {_format_volume(iou.get('intersection_volume'))} · "
                    f"объединение: {_format_volume(iou.get('union_volume'))}"
                )
                st.caption(
                    "IoU = |пересечение| / |объединение|, без выравнивания и масштабирования."
                )
            if cad.get("metric_error"):
                st.error(f"Метрика IoU: {cad['metric_error']}")
            if "candidate_vertices" in cad:
                target = (cad["target_vertices"], cad["target_faces"])
                tab_cad, tab_overlay = st.tabs(["CadQuery", "Наложение"])
                with tab_cad:
                    st.plotly_chart(
                        viewer(
                            cad["candidate_vertices"],
                            cad["candidate_faces"],
                            cad["request_sha"],
                        ),
                        config={"displayModeBar": False, "scrollZoom": True},
                        width="stretch",
                        key="cad-view",
                    )
                with tab_overlay:
                    st.plotly_chart(
                        viewer(
                            cad["candidate_vertices"],
                            cad["candidate_faces"],
                            cad["request_sha"],
                            target=target,
                        ),
                        config={"displayModeBar": False, "scrollZoom": True},
                        width="stretch",
                        key="overlay-view",
                    )
                st.caption("Серый — исходник в нормализованных координатах, синий — CadQuery.")
            st.download_button(
                "Скачать CadQuery STL",
                Path(cad["execution"]["mesh_path"]).read_bytes(),
                file_name="candidate.stl",
                mime="model/stl",
            )
            st.download_button(
                "Скачать STEP",
                Path(cad["execution"]["step_path"]).read_bytes(),
                file_name="candidate.step",
                mime="application/step",
            )
            if (Path(cad["run_dir"]) / "iou.json").is_file():
                st.download_button(
                    "Скачать iou.json",
                    (Path(cad["run_dir"]) / "iou.json").read_bytes(),
                    file_name="iou.json",
                    mime="application/json",
                )
            st.caption(f"Артефакты: {cad['run_dir']}")
        with st.expander("Детали выполнения"):
            st.json(cad["execution"])
    elif not stale and CADQUERY_AVAILABLE:
        st.caption("Кнопка выполнит ровно тот код, который показан выше.")

st.divider()
st.caption(
    "Сохранённый запуск лежит в runs/ui/<время>/: source.stl, llm-images/llm-input.png, "
    "renders/*.png, request.json, response.json, code.py, meta.json."
)
