import * as THREE from "three";
import { OrbitControls } from "./vendor/OrbitControls.js";

const $ = (id) => document.getElementById(id);
const text = (tag, value, cls = "") => {
  const e = document.createElement(tag);
  e.textContent = value;
  e.className = cls;
  return e;
};
const money = (n) => Number(n || 0).toFixed(3);
const percent = (n) => (Number.isFinite(n) ? (n * 100).toFixed(1) + "%" : "—");
const number = (n) =>
  Number.isFinite(n)
    ? n < 0.001 && n !== 0
      ? n.toExponential(2)
      : n.toFixed(4)
    : "—";
const keyLabels = {
  missing: "· не введён",
  unverified: "· не проверен",
  valid: "· подтверждён",
  invalid: "· отклонён OpenRouter (401)",
};
const inputLabels = {
  hybrid: "гибрид",
  images: "изображения",
  none: "без вызова LLM",
};
function inputLabel(candidate) {
  return inputLabels[candidate?.input_mode] || "режим не записан";
}
function evidenceWarning(message) {
  if (message.startsWith("Primitive hypotheses"))
    return "Распознанные поверхности — предварительные гипотезы.";
  if (message.startsWith("Mesh is non-watertight"))
    return "Поверхность незамкнута: возможны открытые сечения, объём не выводится.";
  if (message.startsWith("Section output limits"))
    return "Некоторые сечения передаются с уменьшенной детализацией или пропусками.";
  if (message.startsWith("32 KiB evidence budget"))
    return "Достигнут лимит объёма данных: часть точек контуров опущена. Полную форму отверстий по этим точкам восстановить нельзя.";
  if (message.startsWith("Branching section"))
    return "В сечениях есть разветвления; их топология неоднозначна.";
  return message;
}
let checkingKeyEvent = null;
let state = {},
  selectedAsset = "",
  selectedCandidate = "",
  api = "",
  activeKind = "explain",
  mode = "overlay",
  geometryKey = "",
  pending = false,
  lastNotice = "",
  noticeTimer,
  draftTimer,
  messageKey = "",
  draftAsset = null,
  selectedSnapshot = "",
  actionQueue = [],
  inFlightId = null,
  draftRevision = 0;
const submittedDrafts = new Map();
const draftCache = new Map();
function bridge(type, extra = {}) {
  window.parent.postMessage({ isStreamlitMessage: true, type, ...extra }, "*");
}
function action(actionName, data = {}) {
  if (actionName === "job")
    data = {
      ...data,
      asset_id: selectedAsset,
      parent_id: Object.hasOwn(data, "parent_id")
        ? data.parent_id
        : selectedCandidate || null,
    };
  const event = { id: crypto.randomUUID(), action: actionName, data };
  actionQueue.push(event);
  sendNext();
  return event.id;
}
function sendNext() {
  if (pending || !actionQueue.length) return;
  pending = true;
  const e = actionQueue.shift();
  inFlightId = e.id;
  bridge("streamlit:setComponentValue", { value: e, dataType: "json" });
}
bridge("streamlit:componentReady", { apiVersion: 1 });
function resize() {
  let height = 840;
  try {
    height = window.parent.innerHeight;
  } catch {}
  bridge("streamlit:setFrameHeight", { height: Math.max(320, height - 2) });
  resizeScene();
}
window.addEventListener("resize", resize);
// A stable iframe owns the camera and composer. Render messages update data only.
window.addEventListener("message", (e) => {
  if (e.data?.type !== "streamlit:render") return;
  const args = e.data.args;
  state = args.state || {};
  selectedAsset = args.selected_asset || "";
  selectedCandidate = args.selected_candidate || "";
  api = args.api_url;
  activateDraft(selectedAsset);
  if (args.event_ack?.id === inFlightId) {
    if (args.event_ack.id === checkingKeyEvent) {
      checkingKeyEvent = null;
      $("check-key").disabled = false;
    }
    pending = false;
    inFlightId = null;
    acknowledgeDraft(args.event_ack);
  }
  render(args.geometry);
  if (args.notice && args.notice !== lastNotice) showNotice(args.notice);
  lastNotice = args.notice || "";
  sendNext();
  resize();
});
function readDraft(assetId) {
  try {
    const stored = sessionStorage.getItem("mesh2cad-draft-" + assetId);
    if (stored) return JSON.parse(stored);
  } catch {}
  return draftCache.get(assetId) || { text: "", revision: 0 };
}
function saveDraft(assetId, value, revision) {
  if (!assetId) return;
  const draft = { text: value, revision };
  draftCache.set(assetId, draft);
  try {
    if (value)
      sessionStorage.setItem(
        "mesh2cad-draft-" + assetId,
        JSON.stringify(draft),
      );
    else sessionStorage.removeItem("mesh2cad-draft-" + assetId);
  } catch {}
}
function activateDraft(assetId) {
  if (draftAsset === assetId) return;
  clearTimeout(draftTimer);
  if (draftAsset) saveDraft(draftAsset, $("prompt").value, draftRevision);
  draftAsset = assetId;
  const draft = readDraft(assetId);
  $("prompt").value = draft.text;
  draftRevision = draft.revision;
}
function acknowledgeDraft(ack) {
  const submitted = submittedDrafts.get(ack.id);
  if (!submitted) return;
  submittedDrafts.delete(ack.id);
  if (!ack.success) return;
  // Clear only the exact accepted draft. A newer edit or another asset's
  // composer is never overwritten by an asynchronous acknowledgement.
  if (submitted.assetId === selectedAsset) {
    if (
      submitted.revision !== draftRevision ||
      $("prompt").value !== submitted.raw
    )
      return;
    clearTimeout(draftTimer);
    $("prompt").value = "";
    draftRevision += 1;
    saveDraft(selectedAsset, "", draftRevision);
    action("draft", { text: "", asset_id: selectedAsset });
  } else {
    const stored = readDraft(submitted.assetId);
    if (
      stored.revision === submitted.revision &&
      stored.text === submitted.raw
    ) {
      saveDraft(submitted.assetId, "", stored.revision + 1);
    }
  }
}
function showNotice(message) {
  $("notice").textContent = message;
  $("notice").classList.toggle("error", message.startsWith("Ошибка"));
  $("notice").hidden = false;
  clearTimeout(noticeTimer);
  noticeTimer = setTimeout(() => ($("notice").hidden = true), 12000);
}
$("notice").onclick = () => {
  $("notice").hidden = true;
  action("dismiss");
};

const canvas = $("scene");
let renderer,
  scene,
  camera,
  controls,
  targetMesh,
  candidateMesh,
  grid,
  initialized = false;
try {
  renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  renderer.setClearColor(0, 0);
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  scene = new THREE.Scene();
  camera = new THREE.PerspectiveCamera(37, 1, 0.01, 1000);
  camera.position.set(2.2, -2.5, 1.8);
  camera.up.set(0, 0, 1);
  controls = new OrbitControls(camera, canvas);
  controls.enableDamping = true;
  controls.dampingFactor = 0.09;
  controls.screenSpacePanning = true;
  controls.minDistance = 0.08;
  controls.maxDistance = 70;
  scene.add(new THREE.HemisphereLight(0xeaf3ff, 0x354154, 2.8));
  const light = new THREE.DirectionalLight(0xdce9ff, 3.1);
  light.position.set(3, -4, 6);
  scene.add(light);
  const fill = new THREE.DirectionalLight(0x77a9ff, 1.3);
  fill.position.set(-3, 2, 3);
  scene.add(fill);
  grid = new THREE.GridHelper(4, 24, 0x50627a, 0x2e3a4b);
  grid.rotation.x = Math.PI / 2;
  grid.position.z = -0.55;
  grid.material.transparent = true;
  grid.material.opacity = 0.28;
  scene.add(grid);
  const axes = new THREE.AxesHelper(0.25);
  axes.position.set(-1, -1, -0.545);
  scene.add(axes);
  controls.addEventListener("start", () => {
    document
      .querySelectorAll("[data-camera]")
      .forEach((button) => button.classList.remove("active"));
  });
  controls.addEventListener("end", () => {
    if (!selectedAsset) return;
    try {
      localStorage.setItem(
        "mesh2cad-camera-" + selectedAsset,
        JSON.stringify({
          position: camera.position.toArray(),
          target: controls.target.toArray(),
          up: camera.up.toArray(),
        }),
      );
    } catch {}
  });
  function animate() {
    requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, camera);
  }
  animate();
  initialized = true;
} catch (e) {
  showNotice("Ошибка: 3D недоступен. Включите WebGL в браузере. " + e.message);
}
function resizeScene() {
  if (!renderer) return;
  let r = $("viewport").getBoundingClientRect();
  renderer.setSize(r.width, r.height, false);
  camera.aspect = r.width / Math.max(r.height, 1);
  camera.updateProjectionMatrix();
}
new ResizeObserver(resizeScene).observe($("viewport"));
function meshFrom(data, color, opacity) {
  if (!data?.vertices?.length || !data.faces?.length) return null;
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute(
    "position",
    new THREE.Float32BufferAttribute(data.vertices.flat(), 3),
  );
  geometry.setIndex(data.faces.flat());
  geometry.computeVertexNormals();
  const material = new THREE.MeshStandardMaterial({
    color,
    roughness: 0.48,
    metalness: 0.13,
    flatShading: true,
    transparent: opacity < 1,
    opacity,
    side: THREE.DoubleSide,
    depthWrite: opacity > 0.6,
    polygonOffset: true,
    polygonOffsetFactor: color === 0x89b5f8 ? -1 : 1,
  });
  return new THREE.Mesh(geometry, material);
}
function dispose(mesh) {
  if (mesh) {
    scene.remove(mesh);
    mesh.geometry.dispose();
    mesh.material.dispose();
  }
}
let cameraAsset = "";
function updateGeometry(geometry) {
  if (!initialized) return;
  const key =
    selectedAsset + "|" + selectedCandidate + "|" + (geometry?.frame_id || "");
  if (key === geometryKey) return;
  geometryKey = key;
  dispose(targetMesh);
  dispose(candidateMesh);
  targetMesh = meshFrom(
    geometry?.target,
    0xc3ccd9,
    Number($("target-opacity").value),
  );
  candidateMesh = meshFrom(
    geometry?.candidate,
    0x89b5f8,
    Number($("candidate-opacity").value),
  );
  if (targetMesh) scene.add(targetMesh);
  if (candidateMesh) scene.add(candidateMesh);
  if (targetMesh) {
    const box = new THREE.Box3().setFromObject(targetMesh);
    grid.position.z = box.min.z - 0.025;
    $("geometry-info").textContent =
      geometry.target.faces.length.toLocaleString("ru") +
      " треугольников · нормализованные координаты";
    if (geometry.preview?.target_limited || geometry.preview?.candidate_limited)
      $("geometry-info").textContent += " · облегчённое превью";
  }
  if (cameraAsset !== selectedAsset) {
    cameraAsset = selectedAsset;
    let stored;
    try {
      stored = JSON.parse(
        localStorage.getItem("mesh2cad-camera-" + selectedAsset),
      );
    } catch {}
    if (stored?.position) {
      camera.position.fromArray(stored.position);
      controls.target.fromArray(stored.target);
      camera.up.fromArray(stored.up || [0, 0, 1]);
      controls.update();
    } else preset("iso");
  }
  applyDisplay();
}
function preset(name) {
  document
    .querySelectorAll("[data-camera]")
    .forEach((button) =>
      button.classList.toggle(
        "active",
        name !== "fit" && button.dataset.camera === name,
      ),
    );
  if (!camera) return;
  let box = new THREE.Box3();
  if (targetMesh) box.expandByObject(targetMesh);
  else if (candidateMesh) box.expandByObject(candidateMesh);
  if (box.isEmpty())
    box.setFromCenterAndSize(new THREE.Vector3(), new THREE.Vector3(1, 1, 1));
  const center = box.getCenter(new THREE.Vector3());
  const size = box.getSize(new THREE.Vector3()).length();
  let distance = Math.max(
    (size * 0.8) / Math.tan(THREE.MathUtils.degToRad(camera.fov / 2)),
    1,
  );
  distance *= Math.max(1, 1 / camera.aspect);
  const dirs = {
    front: [0, -1, 0.001],
    side: [1, 0, 0.001],
    top: [0, 0.001, 1],
    iso: [1.5, -2, 1.4],
  };
  let direction =
    name === "fit"
      ? camera.position.clone().sub(controls.target).normalize()
      : new THREE.Vector3(...(dirs[name] || dirs.iso)).normalize();
  camera.up.set(0, 0, 1);
  camera.position.copy(center).add(direction.multiplyScalar(distance));
  controls.target.copy(center);
  camera.near = 0.001;
  camera.far = Math.max(100, distance * 20);
  camera.updateProjectionMatrix();
  controls.update();
}
function applyDisplay() {
  if (targetMesh) {
    targetMesh.visible = mode !== "candidate" && $("target-visible").checked;
    targetMesh.material.opacity = Number($("target-opacity").value);
    targetMesh.material.transparent = targetMesh.material.opacity < 1;
    targetMesh.material.wireframe = $("wireframe").checked;
    targetMesh.material.depthWrite = targetMesh.material.opacity > 0.6;
  }
  if (candidateMesh) {
    candidateMesh.visible = mode !== "target" && $("candidate-visible").checked;
    candidateMesh.material.opacity = Number($("candidate-opacity").value);
    candidateMesh.material.transparent = candidateMesh.material.opacity < 1;
    candidateMesh.material.wireframe = $("wireframe").checked;
    candidateMesh.material.depthWrite = candidateMesh.material.opacity > 0.6;
  }
}
document.querySelectorAll("[data-mode]").forEach(
  (b) =>
    (b.onclick = () => {
      mode = b.dataset.mode;
      document
        .querySelectorAll("[data-mode]")
        .forEach((c) => c.classList.toggle("active", c === b));
      applyDisplay();
    }),
);
document
  .querySelectorAll("[data-camera]")
  .forEach((b) => (b.onclick = () => preset(b.dataset.camera)));
[
  "target-visible",
  "candidate-visible",
  "target-opacity",
  "candidate-opacity",
  "wireframe",
].forEach((id) => ($(id).oninput = applyDisplay));
$("display-button").onclick = () =>
  ($("display-popover").hidden = !$("display-popover").hidden);
function expandScene(expanded) {
  document.body.classList.toggle("scene-expanded", expanded);
  const button = $("expand-scene");
  button.setAttribute("aria-pressed", String(expanded));
  button.textContent = expanded ? "⛶ Свернуть" : "⛶ Развернуть";
  button.title = expanded ? "Вернуть панели (Esc)" : "Развернуть рабочую сцену";
  resizeScene();
}
$("expand-scene").onclick = () =>
  expandScene(!document.body.classList.contains("scene-expanded"));
document.addEventListener("keydown", (event) => {
  if (
    event.key === "Escape" &&
    !document.querySelector("dialog[open]") &&
    document.body.classList.contains("scene-expanded")
  ) {
    expandScene(false);
    $("expand-scene").focus();
  }
});
canvas.onkeydown = (e) => {
  const maps = {
    f: "fit",
    F: "fit",
    0: "iso",
    1: "front",
    2: "side",
    3: "top",
  };
  if (maps[e.key]) {
    e.preventDefault();
    preset(maps[e.key]);
  }
};
function versionLabel(c) {
  return c
    ? "v" +
        ((state.candidates || []).findIndex((x) => x.id === c.id) + 1) +
        " · " +
        c.label
    : "";
}
function selected() {
  return (state.candidates || []).find((c) => c.id === selectedCandidate);
}
function asset() {
  return (state.assets || []).find((a) => a.id === selectedAsset);
}
function renderInputDetails(a, c, settings) {
  const mode = inputLabels[settings.input_mode || "hybrid"] || "гибрид";
  $("input-next-mode").textContent =
    settings.provider === "demo"
      ? "Demo: модель не вызывается. Для внешней модели выбран " + mode + "."
      : "Следующий запрос модели: " + mode + ".";
  $("input-selected-mode").textContent = c
    ? "Выбранная версия: " + inputLabel(c) + ". Настройки не меняют старые версии."
    : "Версия пока не выбрана.";
  const active = (state.jobs || []).find((j) =>
    j.asset_id === a?.id &&
    ["queued", "running", "cancelling"].includes(j.status),
  );
  const preparation = [...(state.jobs || [])].reverse().find((j) =>
    j.asset_id === a?.id && j.kind === "prepare_input",
  );
  const summary = a?.input_evidence;
  const container = $("input-summary");
  container.replaceChildren();
  if (summary) {
    container.append(text("p", "Измерения исходника готовы · " + summary.schema_version));
    const counts = [
      ["Сечения", summary.sections],
      ["Плоские участки", summary.planes],
      ["Гипотезы цилиндров", summary.cylinders],
      ["Точки с нормалями", summary.point_samples],
    ];
    for (const [label, value] of counts)
      container.append(text("p", label + ": " + (value ?? "—")));
    container.append(text("p", "SHA256 исходника: " + summary.target_sha256, "small input-hash"));
  } else {
    container.append(text("p", "Измерения ещё не подготовлены. Гибридный запрос подготовит их автоматически."));
  }
  if (active)
    container.append(text("p", "Задача: " + (stages[active.stage] || stages[active.status] || active.stage), "muted small"));
  else if (preparation?.status === "failed" || preparation?.status === "interrupted")
    container.append(text("p", "Подготовка не завершена: " + (preparation.error || preparation.status), "input-error"));
  else if (preparation?.status === "cancelled")
    container.append(text("p", "Подготовка отменена. Можно повторить локально.", "muted small"));
  $("input-warnings").textContent = (summary?.warnings || []).map(evidenceWarning).join(" ");
  $("prepare-input").disabled = !a || !!active || pending;
  $("prepare-input").textContent = active ? "Задача выполняется…" : "Подготовить локально";
  $("cancel-preparation").hidden = active?.kind !== "prepare_input";
  $("cancel-preparation").disabled = active?.status === "cancelling";
  $("cancel-preparation").dataset.jobId = active?.kind === "prepare_input" ? active.id : "";
  $("download-evidence").hidden = !summary;
  if (summary)
    $("download-evidence").href = api + "/api/assets/" + encodeURIComponent(a.id) + "/evidence";
  else $("download-evidence").removeAttribute("href");
}
$("input-info").onclick = () => openDialog($("input-dialog"));
$("prepare-input").onclick = () => {
  $("prepare-input").disabled = true;
  action("job", { kind: "prepare_input" });
};
$("cancel-preparation").onclick = () => {
  const id = $("cancel-preparation").dataset.jobId;
  if (id) action("cancel", { id });
};
function render(geometry) {
  const a = asset(),
    c = selected(),
    assets = state.assets || [],
    candidates = state.candidates || [],
    settings = state.settings || {},
    budget = state.budget || {};
  if (!checkingKeyEvent)
    $("key-status").textContent = keyLabels[settings.api_key_status] || "· не проверен";
  $("connection").textContent =
    state.connection === "ok" ? "Локальный сервис" : "Сервис недоступен";
  $("connection").classList.toggle("error", state.connection !== "ok");
  $("connection").title = state.connection_error || "API · 127.0.0.1:8765";
  if (state.examples?.length)
    $("example-name").textContent =
      state.examples[0].id === "block"
        ? "Блок · локальный пример"
        : state.examples[0].id === "cylinder"
          ? "Цилиндр · локальный пример"
          : state.examples[0].name;
  $("asset-title").textContent = a?.name || "Новый проект";
  $("asset-tag").textContent = a
    ? (a.format || "mesh").toUpperCase() + " · TARGET"
    : "Нет геометрии";
  $("asset-count").textContent = assets.length;
  const assetKey = assets.map((x) => x.id).join("|") + selectedAsset;
  if ($("assets").dataset.key !== assetKey) {
    $("assets").replaceChildren();
    for (const a of [...assets].reverse()) {
      const b = text(
        "button",
        "",
        "asset-item" + (a.id === selectedAsset ? " active" : ""),
      );
      const info = text("span", "");
      info.append(
        text("span", a.name, "asset-name"),
        text(
          "small",
          (a.format || "mesh").toUpperCase() + " · исходная поверхность",
        ),
      );
      b.append(text("span", "◇", "file-icon"), info);
      b.onclick = () => action("select_asset", { id: a.id });
      $("assets").append(b);
    }
    $("assets").dataset.key = assetKey;
    if (!assets.length)
      $("assets").append(
        text("p", "В проекте пока нет файлов.", "muted small"),
      );
  }
  $("scene-empty").hidden = !!a;
  $("scene-legend").hidden = !a;
  $("legend-version").textContent = versionLabel(c) || "Версия не выбрана";
  $("chat-context").textContent =
    versionLabel(c) || (a ? "Только исходник" : "Выберите геометрию");
  $("input-info").textContent =
    "Ввод: " + (inputLabels[settings.input_mode || "hybrid"] || "гибрид");
  $("input-info").disabled = !a;
  renderInputDetails(a, c, settings);
  updateGeometry(geometry);
  const metrics = c?.metrics || {};
  const defs = [
    ["IoU ↑", percent(metrics.iou), "Объёмное совпадение"],
    ["Chamfer ↓", number(metrics.chamfer), "Квадраты расстояний"],
    [
      "F-score ↑",
      percent(metrics.f_score),
      "Порог " + (metrics.threshold ?? 0.01),
    ],
    ["H95 ↓", number(metrics.h95), "95-й перцентиль"],
  ];
  $("metrics").replaceChildren();
  for (const [name, value, hint] of defs) {
    const el = text("div", "", "metric");
    el.append(
      text("div", name, "metric-name"),
      text("strong", value),
      text("small", c ? hint : "Нет оценки"),
    );
    $("metrics").append(el);
  }
  $("metric-description").textContent =
    "IoU: " +
    (metrics.iou_status || "не вычислен") +
    ". Чем выше IoU и F-score, тем лучше; чем ниже Chamfer и H95, тем лучше. Chamfer — сумма средних квадратов расстояний в двух направлениях. H95 — максимум направленных 95-х перцентилей. " +
    (metrics.n_points
      ? "Выборка " +
        metrics.n_points +
        ", seed " +
        metrics.seed +
        ", порог " +
        metrics.threshold +
        ". "
      : "") +
    "Геометрия не выравнивается независимо. Ошибка объёма: " +
    percent(metrics.volume_relative_error) +
    ". Отсутствующие значения обозначены «—». " +
    (c?.error || "");
  $("version-count").textContent = candidates.length;
  $("history").replaceChildren();
  for (const item of [...candidates].reverse()) {
    const b = text(
      "button",
      "",
      "history-row" + (item.id === selectedCandidate ? " active" : ""),
    );
    b.title =
      (item.status === "valid" ? "Готовая версия" : item.error || item.status) +
      " · " +
      (item.source || "") +
      " · " + inputLabel(item) +
      (item.parent_id ? " · родитель " + item.parent_id : "");
    b.append(
      text("span", item.id === selectedCandidate ? "◉" : "○", "version-node"),
      text(
        "span",
        versionLabel(item) + " · " + inputLabel(item) +
          (item.status !== "valid" ? " · ошибка" : ""),
        "row-title",
      ),
      text("span", percent(item.metrics?.iou), "metric-cell"),
      text("span", number(item.metrics?.chamfer), "metric-cell"),
    );
    b.onclick = () => action("select_candidate", { id: item.id });
    $("history").append(b);
  }
  if (!candidates.length)
    $("history").append(
      text(
        "p",
        a
          ? "Постройте первую CAD-версию для сравнения."
          : "После импорта здесь появятся версии.",
        "empty-history",
      ),
    );
  $("selected-parent").textContent = c?.parent_id
    ? "↳ дочерняя версия"
    : "Исходная ветвь";
  $("predict").disabled = !a;
  $("send").disabled = !a;
  $("reevaluate").disabled = !c || !c.mesh_url;
  $("code-button").disabled = !c;
  $("export-button").disabled = !a;
  $("provider-badge").textContent = (settings.provider || "demo").toUpperCase();
  $("provider-badge").title =
    settings.provider === "demo"
      ? "Локальный offline-провайдер. Ответы детерминированы."
      : "Внешний провайдер";
  $("model-name").textContent =
    settings.provider === "demo"
      ? "Demo · локально, без API"
      : settings.model || "Модель не настроена";
  $("budget").textContent =
    "$" +
    money(budget.known_usd) +
    " / $" +
    Number(budget.limit_usd ?? 0.5).toFixed(2);
  $("budget").title =
    "Резерв $" +
    money(budget.reserved_usd) +
    " · доступно $" +
    money(budget.available_usd) +
    " · неизвестные вызовы " +
    (budget.unknown_calls || 0);
  renderMessages();
  renderJobs();
  if (selectedSnapshot !== selectedCandidate && !$("code-dialog").open) {
    $("code-editor").value = c?.code || "";
    selectedSnapshot = selectedCandidate;
  }
}
function renderMessages() {
  const messages = state.messages || [];
  const key = JSON.stringify(messages) + selectedAsset;
  if (key === messageKey) return;
  messageKey = key;
  const container = $("messages"),
    nearEnd =
      container.scrollHeight - container.scrollTop - container.clientHeight <
      100;
  container.replaceChildren();
  if (!messages.length) {
    const welcome = text("div", "", "chat-welcome");
    welcome.append(
      text("div", "✧", "welcome-icon"),
      text("div", "Исследуйте форму вместе с ассистентом"),
      text(
        "p",
        "Обсудите выбранную версию или опишите изменение. Ответ всегда связан с геометрией, выбранной при отправке.",
      ),
    );
    for (const suggestion of [
      "Какие особенности есть у этой детали?",
      "Чем версия отличается от исходника?",
      "Объясни метрики совпадения",
    ]) {
      const b = text("button", suggestion, "suggestion");
      b.onclick = () => {
        $("prompt").value = suggestion;
        $("prompt").dispatchEvent(new Event("input"));
        $("prompt").focus();
      };
      welcome.append(b);
    }
    welcome.append(
      text(
        "p",
        "В режиме Demo ответы и изменения локальные и детерминированные.",
        "muted small",
      ),
    );
    container.append(welcome);
  }
  for (const m of messages) {
    const article = text("article", "", "chat-message " + m.role);
    const role = text(
      "div",
      m.role === "user"
        ? "ВЫ"
        : m.role === "assistant"
          ? "АССИСТЕНТ"
          : "СИСТЕМА",
      "message-role",
    );
    article.append(role, text("div", m.content, "message-content"));
    if (m.related_candidate_id) {
      const related = (state.candidates || []).find(
        (c) => c.id === m.related_candidate_id,
      );
      const open = text(
        "button",
        "Открыть " + (versionLabel(related) || "созданную версию"),
        "related-version",
      );
      open.onclick = () =>
        action("select_candidate", { id: m.related_candidate_id });
      article.append(open);
    }
    if (m.candidate_id) {
      const c = (state.candidates || []).find((c) => c.id === m.candidate_id);
      article.append(
        text(
          "div",
          "Контекст: " + (versionLabel(c) || m.candidate_id),
          "message-context",
        ),
      );
    }
    container.append(article);
  }
  if (nearEnd || messages.length < 3)
    container.scrollTop = container.scrollHeight;
}
const stages = {
  queued: "В очереди",
  running: "Выполняется",
  cancelling: "Останавливаем…",
  cancelled: "Отменено",
  complete: "Завершено",
  failed: "Ошибка",
  interrupted: "Прервано",
  generate: "Генерация кода",
  execute: "Исполнение CadQuery",
  evaluate: "Оценка геометрии",
  explain: "Подготовка объяснения",
  edit: "Изменение модели",
  predict: "Реконструкция",
  prepare_input: "Локальные измерения геометрии",
  "preparing hybrid evidence": "Подготовка измерений геометрии",
};
function renderJobs() {
  const jobs = state.jobs || [];
  const active = jobs.find((j) =>
    ["queued", "running", "cancelling"].includes(j.status),
  );
  const latest = jobs.at(-1);
  const j = active || latest;
  const area = $("job-status");
  area.hidden = !j;
  if (!j) return;
  area.replaceChildren();
  const description = text("div", stages[j.status] || j.status);
  description.append(
    text(
      "small",
      (stages[j.stage] || j.stage || stages[j.kind] || j.kind) +
        " · " +
        Math.round(j.elapsed_seconds || 0) +
        " с",
    ),
  );
  if (j.error) description.append(text("small", j.error));
  area.append(description);
  if (active) {
    const cancel = text(
      "button",
      j.status === "cancelling" ? "Остановка…" : "Стоп",
    );
    cancel.disabled = j.status === "cancelling";
    cancel.onclick = () => action("cancel", { id: j.id });
    area.append(cancel);
  }
}

$("import").onclick = () => $("file-input").click();
$("file-input").onchange = async (e) => {
  const file = e.target.files[0];
  if (!file) return;
  if (file.size > 25 * 1024 * 1024) {
    showNotice("Ошибка: файл больше 25 МиБ. Выберите меньший mesh.");
    return;
  }
  showNotice("Импортируем " + file.name + "…");
  const reader = new FileReader();
  reader.onload = () =>
    action("upload", { name: file.name, content: reader.result.split(",")[1] });
  reader.readAsDataURL(file);
  e.target.value = "";
};
// Example IDs are discovered through the host, never guessed in a model request.
async function loadExample() {
  action("example", { id: state.examples?.[0]?.id });
}
$("load-example").onclick = loadExample;
$("empty-example").onclick = loadExample;
// New reconstructions start from the same target only, so input-mode A/B
// comparisons do not accidentally inherit the previous run's CAD program.
$("predict").onclick = () => action("job", { kind: "predict", parent_id: null });
document.querySelectorAll("[data-kind]").forEach(
  (b) =>
    (b.onclick = () => {
      activeKind = b.dataset.kind;
      document
        .querySelectorAll("[data-kind]")
        .forEach((c) => c.classList.toggle("active", c === b));
      $("prompt").placeholder =
        activeKind === "edit"
          ? "Опишите изменение выбранной CAD-версии…"
          : "Спросите о геометрии или выбранной версии…";
    }),
);
function send() {
  const raw = $("prompt").value;
  const prompt = raw.trim();
  if (!prompt || !selectedAsset) return;
  if (activeKind === "edit" && !selectedCandidate) {
    showNotice("Ошибка: сначала постройте или выберите CAD-версию.");
    return;
  }
  if (
    [...submittedDrafts.values()].some(
      (draft) =>
        draft.assetId === selectedAsset &&
        draft.revision === draftRevision &&
        draft.raw === raw,
    )
  )
    return;
  clearTimeout(draftTimer);
  const id = action("job", { kind: activeKind, prompt });
  submittedDrafts.set(id, {
    assetId: selectedAsset,
    revision: draftRevision,
    raw,
  });
}
$("send").onclick = send;
$("prompt").onkeydown = (e) => {
  if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
    e.preventDefault();
    send();
  }
};
$("prompt").oninput = () => {
  draftRevision += 1;
  const assetId = selectedAsset,
    value = $("prompt").value;
  saveDraft(assetId, value, draftRevision);
  clearTimeout(draftTimer);
  draftTimer = setTimeout(
    () => action("draft", { text: value, asset_id: assetId }),
    700,
  );
};
$("metrics-help").onclick = () =>
  ($("metric-details").hidden = !$("metric-details").hidden);
function showSettings() {
  const form = $("settings-form"),
    s = state.settings || {},
    b = state.budget || {};
  const models = s.supported_models || [s.model].filter(Boolean);
  form.elements.model.replaceChildren(
    ...models.map((name) => {
      const option = text("option", name);
      option.value = name;
      return option;
    }),
  );
  for (const field of [
    "provider",
    "model",
    "input_mode",
    "reasoning_effort",
    "budget_limit_usd",
    "max_attempts",
  ])
    if (s[field] !== undefined) form.elements[field].value = s[field];
  form.elements.api_key.value = "";
  $("key-status").textContent = keyLabels[s.api_key_status] || "· не проверен";
  $("budget-detail").textContent =
    "Известные расходы $" +
    money(b.known_usd) +
    ". Резерв $" +
    money(b.reserved_usd) +
    ". Доступно $" +
    money(b.available_usd) +
    ". Неопределённые списания остаются в резерве.";
  openDialog($("settings-dialog"));
}
$("settings-button").onclick = showSettings;
$("budget").onclick = showSettings;
$("check-key").onclick = () => {
  const field = $("settings-form").elements.api_key;
  const data = field.value ? { api_key: field.value } : {};
  field.value = "";
  $("check-key").disabled = true;
  $("key-status").textContent = "· проверяем…";
  checkingKeyEvent = action("check_key", data);
};
$("settings-form").onsubmit = (e) => {
  e.preventDefault();
  const form = e.target,
    data = {};
  for (const field of ["provider", "model", "reasoning_effort", "input_mode"])
    data[field] = form.elements[field].value;
  for (const field of ["budget_limit_usd", "max_attempts"])
    data[field] = Number(form.elements[field].value);
  if (form.elements.api_key.value) data.api_key = form.elements.api_key.value;
  form.elements.api_key.value = "";
  action("settings", data);
  $("settings-dialog").close();
};
document
  .querySelectorAll(".close-dialog")
  .forEach((b) => (b.onclick = () => b.closest("dialog").close()));
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape") $("display-popover").hidden = true;
});
$("code-button").onclick = () => {
  const c = selected();
  if (!c) return;
  if (selectedSnapshot !== selectedCandidate) {
    $("code-editor").value = c.code || "";
    selectedSnapshot = selectedCandidate;
  }
  $("download-code").href =
    api + "/api/candidates/" + encodeURIComponent(c.id) + "/code";
  openDialog($("code-dialog"));
};
$("execute").onclick = () => {
  if (!$("code-editor").value.trim()) return;
  action("job", { kind: "execute", code: $("code-editor").value });
  $("code-dialog").close();
};
$("export-button").onclick = () => {
  const c = selected(),
    links = $("export-links");
  links.replaceChildren();
  if (c) {
    for (const [artifact, label, available] of [
      ["step", "STEP · CAD-тело", !!c.step_url],
      ["mesh", "STL · поверхность", !!c.mesh_url],
      ["code", "Python · CadQuery", !!c.code],
      ["comparison", "PNG · сравнение", !!c.comparison_url],
    ]) {
      if (available) {
        const a = text("a", "↓ " + label);
        a.href =
          api + "/api/candidates/" + encodeURIComponent(c.id) + "/" + artifact;
        a.target = "_blank";
        a.rel = "noopener";
        a.download = "";
        links.append(a);
      }
    }
  }
  for (const format of ["json", "csv"]) {
    const a = text("a", "↓ Отчёт · " + format.toUpperCase());
    a.href =
      api +
      "/api/assets/" +
      encodeURIComponent(selectedAsset) +
      "/report?format=" +
      format;
    a.target = "_blank";
    a.rel = "noopener";
    links.append(a);
  }
  openDialog($("export-dialog"));
};
document.querySelectorAll("[data-panel]").forEach(
  (b) =>
    (b.onclick = () => {
      document
        .querySelector(".workspace")
        .classList.remove("show-assets", "show-chat");
      if (b.dataset.panel !== "model")
        document
          .querySelector(".workspace")
          .classList.add("show-" + b.dataset.panel);
      document
        .querySelectorAll("[data-panel]")
        .forEach((c) => c.classList.toggle("active", c === b));
      resizeScene();
    }),
);
resize();

$("reevaluate").onclick = () =>
  action("job", {
    kind: "evaluate",
    n_points: Number($("evaluation-points").value),
    seed: Number($("evaluation-seed").value),
    threshold: Number($("evaluation-threshold").value),
  });
function compatibleMetrics(candidate, anchor, key) {
  if (candidate.id === anchor.id) return true;
  const fields = ["target_sha256", "metric_version", "coordinate_frame"];
  if (key === "chamfer") fields.push("n_points", "seed");
  return fields.every(
    (field) =>
      anchor.metrics?.[field] != null &&
      candidate.metrics?.[field] === anchor.metrics[field],
  );
}
$("selection-policy").onchange = (e) => {
  const key = e.target.value;
  if (key === "manual") return;
  const measured = (state.candidates || []).filter(
    (c) => c.metrics && Object.keys(c.metrics).length,
  );
  const anchor =
    selected()?.metrics && Object.keys(selected().metrics).length
      ? selected()
      : measured.at(-1);
  const available = measured.filter((c) => Number.isFinite(c.metrics?.[key]));
  const candidates = anchor
    ? available.filter((c) => compatibleMetrics(c, anchor, key))
    : [];
  if (!candidates.length) {
    showNotice(
      "Ошибка: нет вычисленных значений с совместимыми параметрами оценки.",
    );
    e.target.value = "manual";
    return;
  }
  candidates.sort((a, b) =>
    key === "iou"
      ? b.metrics.iou - a.metrics.iou
      : a.metrics.chamfer - b.metrics.chamfer,
  );
  const metrics = anchor.metrics;
  const config =
    "метрики " +
    (metrics.metric_version ?? "неизвестны") +
    ", система " +
    (metrics.coordinate_frame ?? "неизвестна") +
    ", target " +
    (metrics.target_sha256?.slice(0, 8) ?? "неизвестен") +
    (key === "chamfer"
      ? ", точки " + metrics.n_points + ", seed " + metrics.seed
      : "");
  showNotice(
    "Выбор по " +
      (key === "iou" ? "IoU" : "Chamfer") +
      ": " +
      config +
      ". Несопоставимых оценок исключено: " +
      (available.length - candidates.length) +
      ".",
  );
  action("select_candidate", { id: candidates[0].id });
};

function openDialog(dialog) {
  dialog.returnFocusTo = document.activeElement;
  dialog.showModal();
}
function closeDialog(dialog) {
  dialog.close();
}
for (const dialog of document.querySelectorAll("dialog")) {
  const title = dialog.querySelector("h2");
  title.id = dialog.id + "-title";
  dialog.setAttribute("aria-labelledby", title.id);
  dialog.addEventListener("cancel", (event) => {
    event.preventDefault();
    closeDialog(dialog);
  });
  dialog.addEventListener("close", () => {
    if (dialog.id === "settings-dialog")
      $("settings-form").elements.api_key.value = "";
    const trigger = dialog.returnFocusTo;
    if (trigger?.isConnected) trigger.focus();
  });
}
document.addEventListener(
  "keydown",
  (event) => {
    const dialog = document.querySelector("dialog[open]");
    if (!dialog) return;
    if (event.key === "Escape") {
      event.preventDefault();
      event.stopPropagation();
      closeDialog(dialog);
      return;
    }
    if (event.key !== "Tab") return;
    const focusable = [
      ...dialog.querySelectorAll(
        'button:not([disabled]), a[href], input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])',
      ),
    ].filter((element) => element.getClientRects().length > 0);
    if (!focusable.length) {
      event.preventDefault();
      return;
    }
    const index = focusable.indexOf(document.activeElement);
    const next =
      index < 0
        ? event.shiftKey
          ? focusable.length - 1
          : 0
        : (index + (event.shiftKey ? -1 : 1) + focusable.length) %
          focusable.length;
    event.preventDefault();
    event.stopPropagation();
    focusable[next].focus();
  },
  true,
);
