"""Streamlit control room. Workers use files, never Streamlit session state."""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

from .dataset import read_catalogue
from .harness import (
    HarnessConfig,
    cancel_run,
    is_active,
    launch_run,
    results_csv,
    start_run,
)
from .observability import configure_logging, tail_log
from .pipeline import DEFAULT_MODEL, REASONING_EFFORTS, api_key_from_environment, preview_payload
from .preview import comparison_figure, evaluate_saved_pair, preview_mesh
from .prompts import DEFAULT_REQUEST, SYSTEM_PROMPT
from .storage import read_json, resolve_artifact, write_json

STATUS = {
    "queued": "В очереди на выполнение",
    "pending": "В очереди",
    "running": "В работе",
    "completed": "Завершён",
    "cancelled": "Остановлен",
    "budget_exhausted": "Лимит исчерпан",
    "failed": "Ошибка",
    "success": "Цель достигнута",
    "exhausted": "Лимит итераций",
    "stalled": "Нет улучшений",
    "unscorable": "IoU неприменим",
}


def _score(value) -> str:
    return "—" if value is None else f"{value:.4f}"


def _best(item: dict) -> dict | None:
    return item["attempts"][item["best_attempt"]] if item["best_attempt"] is not None else None


def _catalogue(manifest_path: Path) -> dict:
    return _catalogue_cached(str(manifest_path), manifest_path.stat().st_mtime_ns)


@st.cache_data(show_spinner=False, max_entries=8)
def _catalogue_cached(path: str, modified: int) -> dict:
    return read_catalogue(Path(path))


def _visible_revision(state: dict) -> tuple:
    return (
        state["status"],
        tuple(
            (
                i["status"],
                i["best_attempt"],
                tuple(
                    (
                        a["status"],
                        (a.get("iou") or {}).get("iou"),
                        (a.get("execution") or {}).get("status"),
                    )
                    for a in i["attempts"]
                ),
            )
            for i in state["items"]
        ),
    )


def _live_status(root: Path, initial_revision: tuple) -> None:
    state = read_json(root / "run.json")
    if _visible_revision(state) != initial_revision:
        st.rerun()
    memory_path = root / "memory.json"
    if memory_path.is_file():
        memory = read_json(memory_path)
        st.caption(
            f"Память worker + CadQuery: {memory.get('rss_mb', 0):.0f} МиБ · "
            f"пик {memory.get('peak_rss_mb', 0):.0f} МиБ · лимит {memory['limit_mb']} МиБ"
        )
    if state["status"] not in {"running", "queued"}:
        st.caption(f"Состояние: {STATUS.get(state['status'], state['status'])}")
        return
    current = state.get("current", {})
    stage = current.get("stage", state["status"])
    elapsed = (
        datetime.now(timezone.utc)
        - datetime.fromisoformat(current.get("started_at", state["updated_at"]))
    ).total_seconds()
    st.caption(f"Текущий этап: {stage} · {current.get('item_id') or '—'} · {max(0, elapsed):.0f} с")


@st.fragment(run_every=2)
def _console(path: Path) -> None:
    st.code(tail_log(path) or "Ожидание событий…", language="log", height=440)
    st.download_button(
        "Скачать project.log",
        lambda: path.read_bytes(),
        file_name="project.log",
        on_click="ignore",
        disabled=not path.is_file(),
    )


@st.cache_data(show_spinner=False, max_entries=12)
def _display_mesh(path: str, modified: int, normalize: bool = False) -> dict:
    return preview_mesh(Path(path), normalize=normalize)


def _comparison(target: Path, candidate: Path | None, key: str, *, normalize_target=False) -> None:
    modes = {
        "Наложение": "overlay",
        "Рядом": "side_by_side",
        "Исходник": "target",
        "CadQuery": "candidate",
    }
    mode = st.radio("Режим preview", list(modes), horizontal=True, key=f"view-mode-{key}")
    target_opacity, candidate_opacity = 0.35, 0.85
    if mode == "Наложение" and candidate:
        left, right = st.columns(2)
        target_opacity = left.slider(
            "Непрозрачность исходника", 0.0, 1.0, 0.35, 0.05, key=f"target-alpha-{key}"
        )
        candidate_opacity = right.slider(
            "Непрозрачность CadQuery", 0.0, 1.0, 0.85, 0.05, key=f"candidate-alpha-{key}"
        )
    target_data = _display_mesh(str(target), target.stat().st_mtime_ns, normalize_target)
    candidate_data = (
        _display_mesh(str(candidate), candidate.stat().st_mtime_ns) if candidate else None
    )
    if mode == "CadQuery" and candidate_data is None:
        st.info("CadQuery-результат ещё не получен.")
        return
    figure = comparison_figure(
        target_data,
        candidate_data,
        mode=modes[mode],
        target_opacity=target_opacity,
        candidate_opacity=candidate_opacity,
        revision=f"{key}-{mode}",
    )
    st.plotly_chart(figure, width="stretch", key=f"mesh-{key}")
    if any(
        data and data["total_faces"] > len(data["faces"]) for data in (target_data, candidate_data)
    ):
        st.caption(
            "Для скорости 3D preview показывает до 40 000 граней. IoU считается по полным STL."
        )


@st.cache_data(show_spinner=False, max_entries=16)
def _evaluate_preview(
    target: str, candidate: str, signature: tuple, normalize_target: bool
) -> dict:
    return evaluate_saved_pair(Path(target), Path(candidate), normalize_target=normalize_target)


def _run_preview(root: Path, item: dict, attempt: dict | None = None) -> None:
    target = root / "items" / item["id"] / "target.stl"
    normalize_target = not target.is_file()
    if normalize_target:
        target = resolve_artifact(root, item["input_path"])
    mesh_path = ((attempt or {}).get("execution") or {}).get("mesh_path")
    candidate = Path(mesh_path) if mesh_path and Path(mesh_path).is_file() else None
    iteration = (attempt or {}).get("iteration", "pending")
    key = f"{root.name}-{item['id']}-{iteration}"
    iou = (attempt or {}).get("iou") or {}
    if candidate and target.is_file():
        signature = (target.stat().st_mtime_ns, candidate.stat().st_mtime_ns)
        if st.button("Пересчитать IoU по STL", key=f"recompute-{key}"):
            try:
                with st.spinner("Объёмное пересечение и объединение полных STL…"):
                    score = _evaluate_preview(
                        str(target), str(candidate), signature, normalize_target
                    )
                st.session_state[f"score-{key}"] = (signature, score)
            except Exception as exc:
                st.error(f"Не удалось пересчитать IoU: {exc}")
        saved = st.session_state.get(f"score-{key}")
        if saved and saved[0] == signature:
            iou = saved[1]
    metrics = st.columns(4)
    metrics[0].metric("IoU выбранной итерации", _score(iou.get("iou")))
    metrics[1].metric("Объём исходника", _score(iou.get("target_volume")))
    metrics[2].metric("Объём CadQuery", _score(iou.get("prediction_volume")))
    missing = ((attempt or {}).get("difference") or {}).get("missing_ratio")
    metrics[3].metric("Missing surface", "—" if missing is None else f"{missing:.1%}")
    if iou:
        st.caption(
            f"Статус IoU: {iou.get('status')} · пересечение: {_score(iou.get('intersection_volume'))} · "
            f"объединение: {_score(iou.get('union_volume'))}. Без выравнивания результата."
        )
    elif not candidate:
        st.caption("Показан исходный STL. Наложение и IoU появятся после выполнения CadQuery.")
    try:
        if target.is_file():
            _comparison(target, candidate, key, normalize_target=normalize_target)
        else:
            st.warning("Исходный STL этого запуска отсутствует на диске.")
    except Exception as exc:
        st.warning(f"Не удалось показать 3D: {exc}")
    if attempt:
        overlay = root / attempt["path"] / "overlay.png"
        if overlay.is_file():
            with st.expander("Карта расхождений · 6 видов", expanded=True):
                st.image(
                    str(overlay),
                    width="stretch",
                    caption="Серый — target, синий — CadQuery, красный — отсутствующая поверхность",
                )
        if attempt.get("overlay_error"):
            st.warning(attempt["overlay_error"])


def _download(path: Path, label: str, key: str) -> None:
    if path.is_file():
        st.download_button(
            label, lambda: path.read_bytes(), file_name=path.name, key=key, on_click="ignore"
        )


def _item_details(root: Path, item: dict) -> None:
    best = _best(item)
    if item.get("error"):
        st.error(item["error"])
    if not item["attempts"]:
        st.info(item.get("stop_reason") or "Агент ещё не обрабатывал эту модель.")
        _run_preview(root, item)
        return
    versions = ["best", "latest"] + list(range(len(item["attempts"])))
    version = st.selectbox(
        "Просмотр итерации",
        versions,
        format_func=lambda v: (
            "Лучшая"
            if v == "best"
            else "Последняя"
            if v == "latest"
            else f"{v + 1} · {item['attempts'][v]['status']}"
        ),
        key=f"attempt-{root.name}-{item['id']}",
    )
    if version == "best":
        iteration = item["best_attempt"] if best else len(item["attempts"]) - 1
    elif version == "latest":
        iteration = len(item["attempts"]) - 1
    else:
        iteration = version
    attempt = item["attempts"][iteration]
    folder = root / attempt["path"]
    st.caption(
        f"Итерация {attempt['iteration']} · {attempt['status']} · "
        f"{item.get('name') or item['source_file']}"
    )
    _run_preview(root, item, attempt)
    rows = [
        {
            "Итерация": a["iteration"],
            "Статус": a["status"],
            "IoU": (a.get("iou") or {}).get("iou"),
            "Missing": (a.get("difference") or {}).get("missing_ratio"),
            "Время, с": a.get("elapsed_seconds"),
            "Лучшая": best is not None and a["iteration"] == best["iteration"],
        }
        for a in item["attempts"]
    ]
    st.dataframe(rows, hide_index=True, width="stretch")
    scored = [row for row in rows if row["IoU"] is not None]
    if scored:
        chart = go.Figure(
            go.Scatter(
                x=[r["Итерация"] for r in scored],
                y=[r["IoU"] for r in scored],
                mode="lines+markers",
                name="IoU",
                line=dict(color="#60a5fa", width=3),
            )
        )
        chart.add_hline(y=st.session_state.get("harness_target_iou", 0.9), line_dash="dot")
        chart.update_layout(
            height=230,
            margin=dict(l=0, r=0, t=10, b=0),
            xaxis=dict(title="Итерация", dtick=1),
            yaxis=dict(title="IoU", range=[0, 1.02]),
        )
        st.plotly_chart(chart, width="stretch", key=f"iou-{root.name}-{item['id']}")
    code, trace, artifacts = st.tabs(
        ["Код и ошибки", "Запрос / ответ", "Файлы"],
        key=f"artifacts-tab-{root.name}-{item['id']}",
        on_change="rerun",
    )
    with st.expander("Исходные виды для агента"):
        image = root / "items" / item["id"] / "target.png"
        if image.is_file():
            st.image(str(image), width="stretch")
    if code.open:
        with code:
            if attempt.get("error"):
                st.code(attempt["error"], language="text")
            st.code(attempt.get("code") or "# Ответ ещё не получен", language="python")
            for name in ("stdout.log", "stderr.log", "traceback.txt"):
                path = folder / "cadquery" / name
                if path.is_file():
                    with st.expander(name):
                        st.code(
                            path.read_text(encoding="utf-8", errors="replace")[-16000:] or "(пусто)"
                        )
    if trace.open:
        with trace:
            for name in ("request.json", "response.json", "execution.json", "iou.json"):
                path = folder / name
                if path.is_file():
                    with st.expander(name):
                        data = read_json(path)
                        st.json(preview_payload(data) if name == "request.json" else data)
                        _download(
                            path, f"Скачать {name}", f"{root.name}-{item['id']}-{iteration}-{name}"
                        )
    if artifacts.open:
        with artifacts:
            for name in (
                "code.py",
                "iou.json",
                "difference.json",
                "overlay.png",
                "attempt.json",
                "cadquery/candidate.stl",
                "cadquery/candidate.step",
            ):
                _download(
                    folder / name,
                    f"Скачать {Path(name).name}",
                    f"artifact-{root.name}-{item['id']}-{iteration}-{name}",
                )
            st.caption(str(folder))


@st.fragment
def _dashboard(root: Path) -> None:
    try:
        state = read_json(root / "run.json")
    except (OSError, ValueError) as exc:
        st.error(f"Не удалось прочитать запуск: {exc}")
        return
    active = is_active(root)
    status = STATUS.get(state["status"], state["status"])
    if state["status"] == "running" and not active:
        status = "Прерван: можно продолжить из checkpoint"
    dataset_name = state["dataset"].get("name") or Path(state["dataset"]["manifest"]).parent.name
    st.markdown(f"**{status}** · `{root.name}` · **{dataset_name}** · `{state['config']['model']}`")
    stop_col, resume_col, info_col = st.columns([1, 1, 3])
    if stop_col.button("Остановить", disabled=not active, key=f"stop-{root.name}"):
        cancel_run(root)
        st.toast("Остановка запрошена. Worker и CadQuery завершатся в течение двух секунд.")
    resume_disabled = active or state["status"] in {"completed", "budget_exhausted"}
    if resume_col.button("Продолжить", disabled=resume_disabled, key=f"resume-{root.name}"):
        try:
            start_run(root, st.session_state.get("harness_effective_key", ""))
            st.rerun()
        except Exception as exc:
            st.error(str(exc))
    live_seconds = state["active_seconds"]
    if active and state["status"] == "running":
        live_seconds += max(
            0,
            (
                datetime.now(timezone.utc) - datetime.fromisoformat(state["updated_at"])
            ).total_seconds(),
        )
    info_col.caption(
        f"Активное время: {live_seconds:.1f} с / {state['config']['max_wall_seconds']:.0f} с · "
        f"причина: {state.get('stop_reason') or '—'}"
    )
    if (root / "cancel.requested").exists() and active:
        st.info("Ожидается остановка текущей операции.")
    if state.get("error"):
        st.error(state["error"])
    summary = state["summary"]
    st.progress(
        summary["finished"] / summary["total"],
        text=f"Обработано {summary['finished']} из {summary['total']} STL",
    )
    columns = st.columns(5)
    for col, label, value in zip(
        columns,
        ["IoU · оценённые", "IoU · вся очередь¹", "Достигли цели", "Токены", "Стоимость, USD"],
        [
            _score(summary["mean_iou"]),
            _score(summary["mean_iou_all"]),
            f"{summary['successes']}/{summary['total']}",
            f"{summary['total_tokens']:,}",
            "—" if summary["cost_usd"] is None else f"${summary['cost_usd']:.4f}",
        ],
        strict=True,
    ):
        col.metric(label, value)
    st.caption(
        f"¹ Неоценённые модели учитываются как 0 только в среднем по всей очереди. "
        f"Валидных программ: {summary['valid_attempts']}/{summary['attempts']} · "
        f"LLM: {summary['llm_calls']} выз., {summary['llm_seconds']:.1f} с · "
        f"CadQuery: {summary['execution_seconds']:.1f} с."
    )
    if summary["cost_missing"] or summary["usage_missing"]:
        st.caption(
            f"Стоимость не сообщена для {summary['cost_missing']} ответов; "
            f"usage отсутствует для {summary['usage_missing']}. Показана только известная стоимость."
        )
    details, queue, logs = st.tabs(
        ["Preview · наложение и IoU", "Очередь и метрики", "Логи"],
        key=f"run-tab-{root.name}",
        on_change="rerun",
    )
    if queue.open:
        with queue:
            st.dataframe(
                [
                    {
                        "ID": item["id"],
                        "STL": item["source_file"],
                        "Категория": item.get("category"),
                        "Split": item["split"],
                        "Статус": STATUS.get(item["status"], item["status"]),
                        "Итерации": len(item["attempts"]),
                        "Лучший IoU": (_best(item) or {}).get("iou", {}).get("iou"),
                        "Причина": item.get("stop_reason"),
                    }
                    for item in state["items"]
                ],
                hide_index=True,
                width="stretch",
            )
            st.download_button(
                "Скачать results.csv",
                results_csv(state),
                file_name="results.csv",
                mime="text/csv",
                key=f"csv-{root.name}",
            )
            _download(root / "run.json", "Скачать checkpoint", f"checkpoint-{root.name}")
            with st.expander("Параметры и происхождение данных"):
                st.json(
                    {
                        "config": state["config"],
                        "dataset": state["dataset"],
                        "prompt_version": state["prompt_version"],
                    }
                )
    if details.open:
        with details:
            selected = st.selectbox(
                "STL-модель",
                list(range(len(state["items"]))),
                format_func=lambda i: (
                    f"{state['items'][i].get('name') or state['items'][i]['source_file']} · "
                    f"{STATUS.get(state['items'][i]['status'], state['items'][i]['status'])}"
                ),
                key=f"item-{root.name}",
            )
            st.session_state["harness_target_iou"] = state["config"]["target_iou"]
            _item_details(root, state["items"][selected])
    if logs.open:
        with logs:
            level = st.selectbox(
                "Уровень", ["Все", "INFO", "WARNING", "ERROR"], key=f"level-{root.name}"
            )
            path = root / "events.jsonl"
            events = []
            if path.is_file():
                tail = tail_log(path).splitlines()
                for line in tail:
                    try:
                        event = json.loads(line)
                        if level == "Все" or event["level"] == level:
                            events.append(event)
                    except ValueError:
                        continue  # The worker may be appending the final line.
                st.code(
                    "\n".join(json.dumps(e, ensure_ascii=False) for e in events) or "Нет событий",
                    language="json",
                    height=360,
                )
                _download(path, "Скачать events.jsonl", f"events-{root.name}")
            else:
                st.info("События появятся после запуска worker.")
    st.caption(f"Артефакты: {root}")


def _catalogue_preview(manifest_path: Path, item: dict) -> None:
    source = resolve_artifact(manifest_path.parent, item["mesh_path"])
    with st.expander("Preview выбранного STL", expanded=True):
        st.markdown(f"**{item['name']}**")
        image = manifest_path.parent / "previews" / f"{item['id']}_aligned.jpg"
        if image.is_file():
            picture, info = st.columns([1, 2])
            picture.image(str(image), width=240)
        else:
            info = st.container()
        with info:
            st.caption(
                f"ID: {item['id']} · категория: {item.get('category') or '—'} · "
                f"источник: {item.get('source') or 'DeepCAD'}"
            )
            volume = item.get("target_is_volume")
            if volume is None:
                volume = item.get("target_watertight")
            st.write(
                "Объёмный IoU: "
                + (
                    "доступен"
                    if volume
                    else "недоступен"
                    if volume is False
                    else "проверяется при запуске"
                )
            )
            if item.get("faces"):
                st.caption(f"Граней: {item['faces']:,}")
            if item.get("source_page"):
                st.link_button("Источник модели", item["source_page"])
            show_3d = st.toggle(
                "Показать 3D STL", value=not image.is_file(), key=f"catalogue-3d-{manifest_path}"
            )
        if show_3d:
            try:
                _comparison(
                    source, None, f"catalogue-{manifest_path}-{item['id']}", normalize_target=True
                )
            except Exception as exc:
                st.warning(f"Не удалось прочитать STL: {exc}")


def render_harness(project_root: Path) -> None:
    configure_logging(project_root / "runs/logs")
    st.title("Mesh2CAD · Harness")
    st.caption("STL → агент → CadQuery → IoU → обратная связь → следующая итерация")
    runs_dir = project_root / "runs" / "harness"
    with st.sidebar:
        # Do not copy a server-side .env secret into a browser widget. A blank
        # override uses the environment key without transmitting it to the UI.
        override = st.text_input(
            "OpenRouter API key (переопределить)", type="password", key="harness_key_override"
        )
        effective_key = override.strip() or api_key_from_environment(project_root)
        st.session_state["harness_effective_key"] = effective_key
        st.caption(
            "Ключ настроен" if effective_key else "Добавьте ключ или OPENROUTER_API_KEY в .env"
        )
        st.caption("Ключ хранится в памяти процесса и не входит в checkpoint.")
        datasets = {
            "DeepCAD": project_root / "data/deepcad/manifest.json",
            "Interior300": project_root / "data/interior300/manifest.json",
        }
        choice = st.selectbox("Датасет", [*datasets, "Другой manifest"], key="harness_dataset")
        if st.session_state.get("harness_last_dataset") != choice:
            if choice in datasets:
                st.session_state["harness_manifest_path"] = str(datasets[choice])
            st.session_state["harness_last_dataset"] = choice
        manifest_text = st.text_input("Manifest датасета", key="harness_manifest_path")
    manifest_path = Path(manifest_text).expanduser()
    catalogue = None
    try:
        catalogue = _catalogue(manifest_path)
    except (OSError, ValueError, KeyError) as exc:
        st.warning(f"Датасет недоступен: {exc}")
        st.code("uv run python -m mesh2cad.dataset --output data/deepcad", language="bash")
    runs = sorted(runs_dir.glob("*/run.json"), reverse=True)
    if st.session_state.pop("_harness_show_workspace", False):
        st.session_state["harness_workspace"] = "Запущенные · preview"
    workspace, setup, dataset, history, console = st.tabs(
        [
            "Запущенные · preview",
            "Новый запуск",
            "Датасет · каталог STL",
            "История",
            "Консоль проекта",
        ],
        default="Запущенные · preview" if runs else "Новый запуск",
        key="harness_workspace",
        on_change="rerun",
    )
    if setup.open:
        with setup:
            if catalogue:
                ready = [item for item in catalogue["items"] if item.get("status") == "ready"]
                counts = Counter(item["split"] for item in ready)
                st.caption(
                    f"{catalogue['name']} · {len(ready)} STL · "
                    + (
                        f"dev {counts['dev']} / test {counts['test']}"
                        if counts["dev"] or counts["test"]
                        else "без официального разбиения на dev/test"
                    )
                )
                splits = [s for s in ("dev", "test") if counts[s]] + ["all"]
                split = st.radio("Split", splits, horizontal=True, key=f"split-{manifest_path}")
                available = [item for item in ready if split == "all" or item["split"] == split]
                categories = sorted({item["category"] for item in ready if item.get("category")})
                if categories:
                    chosen_categories = st.multiselect(
                        "Категории", categories, key=f"categories-{manifest_path}"
                    )
                    if chosen_categories:
                        available = [i for i in available if i.get("category") in chosen_categories]
                only_volumes = st.checkbox(
                    "Только модели для объёмного IoU",
                    value=catalogue["format"] == "interior300",
                    key=f"volumes-{manifest_path}",
                )
                if only_volumes:
                    available = [
                        i
                        for i in available
                        if (
                            i["target_is_volume"]
                            if i.get("target_is_volume") is not None
                            else i.get("target_watertight")
                        )
                        is not False
                    ]
                st.caption(f"Доступно с текущими фильтрами: {len(available)} STL")
                by_id = {item["id"]: item for item in available}
                selected = st.multiselect(
                    "Модели для запуска (пусто — отфильтрованный список)",
                    list(by_id),
                    format_func=lambda key: f"{by_id[key]['name']} · {key}",
                    key=f"models-{manifest_path}-{split}",
                )
                if available:
                    preview_ids = selected or list(by_id)
                    preview_id = st.selectbox(
                        "Preview модели",
                        preview_ids,
                        format_func=lambda k: by_id[k]["name"],
                        key=f"catalogue-preview-{manifest_path}-{split}",
                    )
                    _catalogue_preview(manifest_path, by_id[preview_id])
                else:
                    st.info("Нет моделей с выбранными фильтрами.")
                with st.form("new-harness-run"):
                    col1, col2, col3 = st.columns(3)
                    model = col1.text_input("Модель OpenRouter", DEFAULT_MODEL)
                    iterations = col2.number_input("Итераций на STL", 1, 30, 5)
                    target_iou = col3.number_input("Целевой IoU", 0.0, 1.0, 0.9, 0.01)
                    limit = col1.number_input(
                        "Максимум STL в запуске",
                        1,
                        max(1, len(available)),
                        min(3, max(1, len(available))),
                    )
                    patience = col2.number_input("Остановка без улучшений, итераций", 1, 30, 3)
                    effort = col3.selectbox("Reasoning effort", REASONING_EFFORTS, index=3)
                    with st.expander("Лимиты и промпты"):
                        budget1, budget2, budget3 = st.columns(3)
                        tokens = budget1.number_input(
                            "Бюджет токенов на запуск", 1, 10_000_000, 100_000
                        )
                        wall = budget2.number_input("Активное время, минут", 1, 1440, 30)
                        max_tokens = budget3.number_input(
                            "Max output tokens / вызов", 1, 100_000, 4096
                        )
                        request_timeout = budget1.number_input("HTTP timeout, с", 1, 900, 180)
                        execution_timeout = budget2.number_input("CadQuery timeout, с", 1, 600, 60)
                        retries = budget3.number_input("Повторы временных ошибок API", 0, 5, 2)
                        memory_limit = budget1.number_input(
                            "Лимит памяти worker + CadQuery, МиБ", 256, 8192, 2048, step=256
                        )
                        min_improvement = st.number_input(
                            "Минимальное улучшение IoU", 0.0, 1.0, 0.001, step=0.001, format="%.3f"
                        )
                        interior = catalogue["format"] == "interior300"
                        default_system = (
                            SYSTEM_PROMPT.replace(
                                "a mechanical part", "a piece of furniture or an interior object"
                            )
                            if interior
                            else SYSTEM_PROMPT
                        )
                        default_request = (
                            "Reconstruct the complete furniture or interior object from the contact sheet. "
                            "Preserve all visible parts, supports, openings and their proportions."
                            if interior
                            else DEFAULT_REQUEST
                        )
                        system_prompt = st.text_area("System prompt", default_system, height=220)
                        request = st.text_area("Задача агента", default_request)
                        st.caption(
                            "Бюджет токенов проверяется по usage до следующего вызова; "
                            "текущий ответ может превысить остаток. Таймер не включает паузы."
                        )
                    submitted = st.form_submit_button(
                        "Запустить оркестратор", type="primary", disabled=not available
                    )
                if submitted:
                    root = None
                    try:
                        if not effective_key:
                            raise ValueError("Добавьте OpenRouter API key в боковой панели")
                        config = HarnessConfig(
                            model=model,
                            max_iterations=iterations,
                            target_iou=target_iou,
                            patience=patience,
                            reasoning_effort=effort,
                            max_tokens=max_tokens,
                            total_token_budget=tokens,
                            max_wall_seconds=wall * 60,
                            request_timeout=request_timeout,
                            execution_timeout=execution_timeout,
                            max_retries=retries,
                            min_improvement=min_improvement,
                            memory_limit_mb=memory_limit,
                            system_prompt=system_prompt,
                            request=request,
                        )
                        with st.spinner("Проверяю STL и сохраняю снимок входных данных…"):
                            root, reused = launch_run(
                                manifest_path,
                                runs_dir,
                                config,
                                effective_key,
                                split=split,
                                item_ids=selected or list(by_id),
                                limit=limit,
                            )
                        st.session_state["harness_selected_run"] = str(root)
                        st.session_state["_harness_show_workspace"] = True
                        if reused:
                            st.toast("Этот запуск уже существует — открыл его")
                        st.rerun()
                    except Exception as exc:
                        st.error(str(exc))
                        if root and not is_active(root):
                            state = read_json(root / "run.json")
                            state.update(status="failed", error=str(exc), stop_reason="start_error")
                            write_json(root / "run.json", state)
    if dataset.open:
        with dataset:
            if catalogue:
                source = catalogue.get("source", {})
                st.write(f"**{catalogue['name']}** · {len(catalogue['items'])} STL")
                st.write(
                    f"Источник: `{source.get('repo') or ', '.join(source.get('sources', [])) or source.get('path', 'local')}`"
                )
                if source.get("revision"):
                    st.caption(f"Revision: {source['revision']}")
                st.dataframe(
                    [
                        {
                            "ID": item["id"],
                            "STL": item["source_file"],
                            "Категория": item.get("category"),
                            "Источник": item.get("source"),
                            "Split": item["split"],
                            "Статус": item["status"],
                            "Файл на диске": (
                                manifest_path.parent / item.get("mesh_path", "missing")
                            ).is_file(),
                            "Замкнутость": item.get("target_watertight"),
                            "Пригоден для IoU": item.get("target_is_volume"),
                        }
                        for item in catalogue["items"]
                    ],
                    hide_index=True,
                    width="stretch",
                )
                st.caption(
                    "Перед запуском проверяются SHA-256 STL. Старый кеш изображений не нужен: "
                    "виды 1536×1024 строятся текущим рендерером."
                )
    runs = sorted(runs_dir.glob("*/run.json"), reverse=True)
    if history.open:
        with history:
            history_rows = []
            for path in runs:
                try:
                    state = read_json(path)
                    history_rows.append(
                        {
                            "Запуск": path.parent.name,
                            "Датасет": state["dataset"].get("name")
                            or Path(state["dataset"]["manifest"]).parent.name,
                            "Модель": state["config"]["model"],
                            "Статус": STATUS.get(state["status"], state["status"]),
                            "STL": state["summary"]["total"],
                            "Средний IoU": state["summary"]["mean_iou"],
                            "Токены": state["summary"]["total_tokens"],
                        }
                    )
                except (OSError, ValueError, KeyError):
                    continue
            st.dataframe(
                history_rows, hide_index=True, width="stretch"
            ) if history_rows else st.info("Сохранённых запусков пока нет.")
    if runs and workspace.open:
        with workspace:
            choices = [str(path.parent) for path in runs]
            if st.session_state.get("harness_selected_run") not in choices:
                st.session_state["harness_selected_run"] = choices[0]
            selected_root = st.selectbox(
                "Текущий запуск",
                choices,
                format_func=lambda p: Path(p).name,
                key="harness_selected_run",
            )
            initial = read_json(Path(selected_root) / "run.json")
            st.fragment(run_every=2 if is_active(Path(selected_root)) else None)(_live_status)(
                Path(selected_root), _visible_revision(initial)
            )
            _dashboard(Path(selected_root))
    elif not runs and workspace.open:
        with workspace:
            st.info(
                "Создайте запуск во вкладке «Новый запуск». Здесь появятся его STL, наложение и IoU."
            )

    if console.open:
        with console:
            st.caption("Все компоненты проекта · консоль сервера и runs/logs/project.log")
            _console(project_root / "runs/logs/project.log")
