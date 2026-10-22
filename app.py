"""Agent harness and a fully visible manual STL reconstruction pipeline."""

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
    OVERLAY_LEGEND,
    OVERLAY_VERSION,
    VIEW_NAMES,
    VIEW_SIZE,
    compose_views,
    load_mesh,
    mesh_difference,
    mesh_stats,
    normalize_mesh,
    render_llm_images,
)
from mesh2cad.observability import configure_logging, timed_stage
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
from mesh2cad.preview import evaluate_saved_pair, render_saved_overlay, solid_overlay_figure
from mesh2cad.prompts import DEFAULT_REQUEST, SYSTEM_PROMPT, overlay_repair_prompt

ROOT = Path(__file__).resolve().parent
configure_logging(ROOT / "runs/logs")
DATASET_DIR = ROOT / "data" / "deepcad"
RUNS_DIR = ROOT / "runs" / "ui"
CADQUERY_AVAILABLE = importlib.util.find_spec("cadquery") is not None

st.set_page_config(page_title="Mesh2CAD · Harness", layout="wide", initial_sidebar_state="expanded")
mode = st.sidebar.radio(
    "Режим", ["Оркестратор", "Оркестратор 2", "Ручной pipeline"], key="app_mode"
)
if mode in {"Оркестратор", "Оркестратор 2"}:
    from mesh2cad.harness_ui import render_harness

    with timed_stage("ui.render", view="harness"):
        if mode == "Оркестратор 2":
            render_harness(ROOT, variant="vision2")
        else:
            render_harness(ROOT)
    st.stop()

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


def _mesh_trace(
    vertices,
    faces,
    *,
    color: str,
    opacity: float,
    name: str,
    flat: bool = False,
):
    lighting = (
        {"ambient": 1.0, "diffuse": 0.0, "specular": 0.0, "roughness": 1.0}
        if flat
        else {"ambient": 0.45, "diffuse": 0.75, "specular": 0.15, "roughness": 0.7}
    )
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
        lighting=lighting,
        lightposition={"x": 100, "y": 200, "z": 100},
        name=name,
    )


def viewer(
    vertices,
    faces,
    revision: str,
    target: tuple | None = None,
    missing: tuple | None = None,
):
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
            opacity=0.9,
            name="CadQuery" if target is not None else "STL",
        )
    )
    if missing is not None and len(missing[1]):
        traces.append(
            _mesh_trace(
                missing[0],
                missing[1],
                color="#ff2020",
                opacity=1.0,
                name="Не хватает",
                flat=True,
            )
        )
    figure = go.Figure(data=traces)
    figure.update_layout(
        height=560,
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
        paper_bgcolor="rgba(0,0,0,0)",
        showlegend=target is not None or missing is not None,
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


def refresh_overlay(state: dict, target_path: Path, candidate_path: Path) -> None:
    folder = Path(state["run_dir"])
    render_saved_overlay(target_path, candidate_path, folder)
    state["overlay_sheet"] = (folder / "overlay-solid.png").read_bytes()
    state["overlay_version"] = OVERLAY_VERSION
    state["overlay_parts_path"] = str(folder / "overlay-solid.npz")


def execute_result(result: dict, data: dict, source_sha: str, base_request_sha: str) -> dict:
    """Run the current code, compute IoU, and build the red missing overlay."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if result.get("run_dir"):
        cad_dir = Path(result["run_dir"]) / "cadquery"
    else:
        cad_dir = RUNS_DIR / f"{stamp}-cadquery"
    state = {
        "status": "execution_error",
        "error": None,
        "execution": {},
        "run_dir": str(cad_dir),
        "source_sha": source_sha,
        "request_sha": result["request_sha"],
        "base_request_sha": base_request_sha,
    }
    try:
        execution = execute_code(result["code"], cad_dir, timeout=60)
    except Exception as exc:  # noqa: BLE001 - keep the failure visible
        state["error"] = f"{type(exc).__name__}: {exc}"
        return state
    state["execution"] = execution
    state["status"] = execution["status"]
    state["error"] = execution.get("error")
    try:
        (cad_dir / "execution.json").write_text(
            json.dumps(execution, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    except OSError:
        pass
    if execution["status"] != "valid":
        return state
    try:
        target = load_mesh(data["normalized_stl"])
        candidate = load_mesh(execution["mesh_path"])
        state["candidate_vertices"] = candidate.vertices
        state["candidate_faces"] = candidate.faces
        target_path = cad_dir / "normalized-target.stl"
        target_path.unlink(missing_ok=True)
        target_path.write_bytes(data["normalized_stl"])
        iou = evaluate_saved_pair(target_path, Path(execution["mesh_path"]))
        state["iou"] = iou
        try:
            (cad_dir / "iou.json").write_text(
                json.dumps(iou, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                encoding="utf-8",
            )
        except OSError:
            pass
    except Exception as exc:  # noqa: BLE001 - show the failure
        state["metric_error"] = f"{type(exc).__name__}: {exc}"
        return state
    try:
        difference = mesh_difference(target, candidate)
        state["difference"] = difference
        state["difference_summary"] = {
            key: difference[key]
            for key in (
                "threshold",
                "missing_faces",
                "covered_faces",
                "total_faces",
                "missing_ratio",
                "max_distance",
            )
        }
        try:
            (cad_dir / "difference.json").write_text(
                json.dumps(
                    state["difference_summary"],
                    ensure_ascii=False,
                    indent=2,
                    allow_nan=False,
                )
                + "\n",
                encoding="utf-8",
            )
        except OSError:
            pass
        refresh_overlay(state, target_path, Path(execution["mesh_path"]))
        (cad_dir / "overlay.png").write_bytes(state["overlay_sheet"])
    except Exception as exc:  # noqa: BLE001 - keep IoU visible
        state["difference_error"] = f"{type(exc).__name__}: {exc}"
    return state


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
                "base_request_sha": request_sha,
                "prompt_mode": "initial",
                "iteration": 1,
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
    result_base_sha = result.get("base_request_sha", result["request_sha"])
    if result["source_sha"] != source_sha:
        st.warning("Ответ относится к другому STL. Загрузите тот же файл или отправьте заново.")
    elif result_base_sha != request_sha:
        st.warning("Ответ относится к другому состоянию полей. Отправьте запрос заново.")
    if result.get("prompt_mode") == "improve":
        st.caption(f"Улучшение · итерация {result.get('iteration', 1)}")
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
    base_sha = result.get("base_request_sha", result["request_sha"])
    stale = result["source_sha"] != source_sha or base_sha != request_sha
    if stale:
        st.warning("Код относится к другому STL или запросу. Отправьте запрос заново.")
    if not CADQUERY_AVAILABLE:
        st.warning("CadQuery не установлен. Выполните `uv sync --extra dev`.")
    if st.button(
        "Выполнить в CadQuery и посчитать IoU",
        type="primary",
        disabled=stale or not CADQUERY_AVAILABLE,
    ):
        with st.spinner("CadQuery в песочнице…"):
            st.session_state["cad"] = execute_result(result, data, source_sha, request_sha)

    cad = st.session_state.get("cad")
    if cad and (
        cad.get("source_sha") != source_sha
        or cad.get("base_request_sha", cad.get("request_sha")) != request_sha
    ):
        st.session_state.pop("cad", None)
        cad = None
    if cad:
        if cad["status"] != "valid":
            st.error(f"CadQuery: {cad['status']}")
            if cad.get("error"):
                st.code(cad["error"], language=None)
        else:
            if cad.get("overlay_version") != OVERLAY_VERSION:
                try:
                    target_path = Path(cad["run_dir"]) / "normalized-target.stl"
                    if not target_path.is_file():
                        target_path.write_bytes(data["normalized_stl"])
                    with st.spinner("Обновляю цветовую схему: совпадение, лишнее, недостаток…"):
                        refresh_overlay(cad, target_path, Path(cad["execution"]["mesh_path"]))
                except Exception as exc:
                    cad.pop("overlay_sheet", None)
                    cad["overlay_error"] = str(exc)
            st.success(
                f"Выполнено за {cad['execution']['elapsed_seconds']:.2f} с · "
                f"изоляция: {cad['execution'].get('sandbox', 'restricted')}"
            )
            iou = cad.get("iou")
            if iou:
                iou_col, target_col, cad_col, missing_col = st.columns(4)
                value = iou.get("iou")
                summary = cad.get("difference_summary") or {}
                iou_col.metric("IoU", "—" if value is None else f"{value:.4f}")
                target_col.metric("Объём исходника", _format_volume(iou.get("target_volume")))
                cad_col.metric("Объём CadQuery", _format_volume(iou.get("prediction_volume")))
                ratio = summary.get("missing_ratio")
                missing_col.metric(
                    "Не хватает",
                    "—" if ratio is None else f"{float(ratio):.1%}",
                )
                st.caption(
                    f"Статус: `{iou.get('status')}` · "
                    f"пересечение: {_format_volume(iou.get('intersection_volume'))} · "
                    f"объединение: {_format_volume(iou.get('union_volume'))}"
                )
                st.caption(
                    "IoU = |пересечение| / |объединение|, без выравнивания и масштабирования. "
                    "Красным — часть target, которой нет в CadQuery."
                )
            if cad.get("metric_error"):
                st.error(f"Метрика IoU: {cad['metric_error']}")
            if cad.get("difference_error"):
                st.error(f"Контрастная схема: {cad['difference_error']}")
            if "candidate_vertices" in cad:
                difference = cad.get("difference")
                if difference:
                    target = (
                        difference["covered_vertices"],
                        difference["covered_triangles"],
                    )
                    missing = (
                        difference["missing_vertices"],
                        difference["missing_triangles"],
                    )
                else:
                    target = (data["normalized_vertices"], data["normalized_faces"])
                    missing = None
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
                    if cad.get("overlay_parts_path"):
                        st.plotly_chart(
                            solid_overlay_figure(Path(cad["overlay_parts_path"])),
                            width="stretch",
                            key="overlay-view",
                        )
                    else:
                        st.plotly_chart(
                            viewer(
                                cad["candidate_vertices"],
                                cad["candidate_faces"],
                                cad["request_sha"],
                                target=target,
                                missing=missing,
                            ),
                            config={"displayModeBar": False, "scrollZoom": True},
                            width="stretch",
                            key="overlay-view",
                        )
                st.caption(
                    OVERLAY_LEGEND
                    if cad.get("overlay_version") == OVERLAY_VERSION
                    else "Архивное наложение: серый — исходник, синий — весь CAD."
                )
                if cad.get("overlay_sheet"):
                    st.markdown("**Контрастная схема для улучшения**")
                    st.image(
                        cad["overlay_sheet"],
                        width="stretch",
                        caption="overlay.png · 1536×1024 · 6 видов",
                    )
                    st.caption(
                        OVERLAY_LEGEND
                        if cad.get("overlay_version") == OVERLAY_VERSION
                        else "Архивная схема: серый — покрытая поверхность, синий — весь CAD, красный — недостающая поверхность."
                    )
                    st.download_button(
                        "Скачать overlay.png",
                        cad["overlay_sheet"],
                        file_name="overlay.png",
                        mime="image/png",
                    )
                if cad.get("overlay_error"):
                    st.error(f"Контрастная схема: {cad['overlay_error']}")
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

    # -----------------------------------------------------------------------
    # 6. Improve: same image and pipeline, previous code appended to the prompt
    # -----------------------------------------------------------------------
    st.divider()
    st.subheader("6 · Улучшение")
    st.caption(
        "В улучшение уходит не исходный лист, а контрастная схема overlay.png: "
        "зелёный — совпадение, синий — лишнее за пределами mesh, красный — чего не хватает. "
        "В system prompt добавляется текущий CadQuery-код."
    )
    overlay = cad.get("overlay_sheet") if cad else None
    if result.get("prompt_mode") == "improve":
        st.caption(f"Текущая итерация улучшения: {result.get('iteration', 1)}")
        with st.expander("Промпт, который был отправлен"):
            st.code(result.get("repair_prompt", ""), language="text")
    next_prompt = overlay_repair_prompt(st.session_state["system_prompt"], result["code"])
    with st.expander("Промпт следующего улучшения"):
        st.code(next_prompt, language="text")
    if not overlay:
        if cad and cad.get("status") == "valid":
            st.warning(
                "Не удалось построить контрастную схему: "
                + str(cad.get("difference_error") or cad.get("overlay_error") or "нет overlay")
            )
        else:
            st.warning("Сначала выполните CadQuery, чтобы получить контрастную схему.")
    improve_disabled = stale or not CADQUERY_AVAILABLE or not api_key.strip() or not overlay
    if st.button(
        "Улучшить",
        type="primary",
        disabled=improve_disabled,
    ):
        with st.spinner("Отправляю контрастную схему на улучшение…"):
            try:
                repair_system = overlay_repair_prompt(
                    st.session_state["system_prompt"], result["code"]
                )
                repair_payload = build_payload(
                    model=st.session_state["model"],
                    system_prompt=repair_system,
                    request=st.session_state["request_text"],
                    images=[("contrast-overlay", overlay)],
                    extents=normalized_stats["extents"],
                    reasoning_effort=st.session_state["reasoning_effort"],
                )
                response = call_openrouter(repair_payload, api_key)
                code = extract_code(response)
                run_dir = save_run(
                    RUNS_DIR,
                    source_name=source_name,
                    source_bytes=source_bytes,
                    images={"contrast-overlay": overlay},
                    extra_images=images,
                    payload=repair_payload,
                    response=response,
                    code=code,
                )
                new_result = {
                    "response": response,
                    "code": code,
                    "run_dir": str(run_dir),
                    "source_sha": source_sha,
                    "request_sha": payload_digest(repair_payload),
                    "base_request_sha": request_sha,
                    "prompt_mode": "improve",
                    "iteration": result.get("iteration", 1) + 1,
                    "repair_prompt": repair_system,
                }
                st.session_state["result"] = new_result
                st.session_state["result_error"] = None
                st.session_state["cad"] = execute_result(new_result, data, source_sha, request_sha)
                st.rerun()
            except Exception as exc:  # noqa: BLE001 - show the failure
                st.session_state["result_error"] = str(exc)
                st.rerun()

st.divider()
st.caption(
    "Сохранённый запуск лежит в runs/ui/<время>/: source.stl, llm-images/llm-input.png, "
    "renders/*.png, request.json, response.json, code.py, meta.json."
)
