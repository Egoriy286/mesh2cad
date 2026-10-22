# STL → LLM → CadQuery

Минимальное приложение: импорт STL, просмотр, прозрачная подготовка данных для LLM,
выполнение полученного CadQuery-кода и объёмный IoU. Всё, что уходит в модель и
возвращается из неё, видно на экране.

## Что осталось

```text
app.py                    Streamlit: импорт STL, 3D-просмотр, pipeline
mesh2cad/geometry.py      загрузка STL, нормализация, RGB/depth 512×512, IoU
mesh2cad/pipeline.py      точный запрос к OpenRouter, разбор ответа, артефакты
mesh2cad/prompts.py       system prompt и запрос по умолчанию
mesh2cad/executor.py      запуск кода в macOS Seatbelt
mesh2cad/_worker.py       CadQuery worker: r -> STEP/STL
mesh2cad/dataset.py       воспроизводимая подготовка DeepCAD
mesh2cad/storage.py       атомарная запись JSON
data/deepcad/             локальный датасет (110 STL, manifest)
tests/                    проверки без сетевых вызовов
```

Удалены внешние репозитории `cadfather`, `cadrille`, `DeepCAD`, старые FastAPI,
кастомный Three.js-компонент, CLI/harness, docs и тесты прежней рабочей среды.
Исторические `runs/` и остальные каталоги `data/` не удалялись.

## Запуск

```sh
uv sync --extra dev
.venv/bin/streamlit run app.py
```

Если используется отдельное окружение проекта:

```sh
UV_PROJECT_ENVIRONMENT=.venv-mesh2cad uv sync --extra dev
.venv-mesh2cad/bin/streamlit run app.py
```

Откройте `http://localhost:8501`.

## Экран

1. **Импорт и просмотр.** Загрузите STL или выберите файл из `data/deepcad/meshes`.
   Слева интерактивная 3D-модель, справа размеры и статистика.
2. **Pipeline.** Находится сразу под просмотром:
   - **1 · Нормализация** — центр bounding box в 0, длиннейшая сторона в 1;
     матрица, границы и normalized STL показаны полностью.
   - **2 · Изображения для LLM** — три вида: top (+Z), left (−X), right (+X).
     Для каждого есть RGB и карта глубины **512 × 512**, фон чёрный; в depth
     ближе — светлее, дальше — темнее. Все шесть тайлов плюс один общий
     contact sheet **1536 × 1024** (3 колонки, сверху RGB, снизу depth) показаны
     и скачиваются. В LLM уходит именно один общий `llm-input.png`.
   - **3 · Ввод** — редактируемые system prompt и запрос пользователя, модель,
     reasoning effort. Ниже показан точный JSON запроса; base64 сокращён только
     в предпросмотре, в скачиваемом `request.json` он полный.
   - **4 · Ответ** — кнопка отправки в OpenRouter, сырой ответ, usage и извлечённый
     CadQuery-код.
   - **5 · CadQuery и IoU** — кнопка выполняет ровно показанный код в отдельном
     процессе под macOS Seatbelt. Результат показывается отдельно и в наложении на
     исходник; считаются объёмы и **IoU = |пересечение| / |объединение|** без
     выравнивания и масштабирования. STL/STEP и `iou.json` можно скачать.

Каждый успешный вызов сохраняется в `runs/ui/<время>-<id>/`:

```text
source.stl
llm-images/
  llm-input.png   # один contact sheet 1536×1024, уходит в LLM
renders/
  top-rgb.png     # отдельные тайлы 512×512
  top-depth.png
  left-rgb.png
  left-depth.png
  right-rgb.png
  right-depth.png
request.json      # точный запрос с base64 одного llm-input.png
response.json     # сырой ответ
code.py
meta.json         # SHA-256, модель, usage, время
cadquery/
  candidate.stl   # результат CadQuery
  candidate.step
  iou.json        # IoU, объёмы, статус применимости
  execution.json  # статус, пути, время выполнения
```

## Ключ OpenRouter

Ключ читается из `OPENROUTER_API_KEY` или из локального `.env`:

```sh
OPENROUTER_API_KEY=...
```

`.env` добавлен в `.gitignore`. В интерфейсе ключ скрыт, хранится только в памяти
сессии и не записывается в `runs/`.

## CadQuery

Код модели выполняется только после ответа LLM и только в отдельном процессе под
macOS Seatbelt: сеть, домашний каталог и запись за пределы папки запуска закрыты.
На других ОС выполнение возвращает `sandbox_error` и не запускается без песочницы.

Worker принимает переменную `r` типа `cq.Workplane`, проверяет валидность и
положительный объём, затем экспортирует `candidate.step` и `candidate.stl`.
IoU считается через `manifold3d`; если тела не замкнуты или boolean-операция
недоступна, в интерфейсе показывается причина, а не выдуманный ноль.

## Датасет

`data/deepcad/` — 10 dev + 100 test STL из
[`maksimko123/deepcad_test_mesh`](https://huggingface.co/datasets/maksimko123/deepcad_test_mesh)
с закреплённой ревизией, manifest, нормализованными mesh и кешем видов.
Каталог `data/` локальный и не коммитится. Пересобрать:

```sh
.venv/bin/python -m mesh2cad.dataset --output data/deepcad
```

`data/interiors/` и `data/replica/` оставлены как есть, приложение их не использует.

## Проверки

```sh
.venv/bin/python -m pytest
.venv/bin/ruff check app.py mesh2cad tests
```

Тесты проверяют нормализацию, шесть тайлов 512×512 (RGB + depth, чёрный фон),
общий sheet 1536×1024, структуру запроса, извлечение кода, сохранение артефактов,
настоящий CadQuery-экспорт в песочнице и IoU на совпадающих, сдвинутых,
разъединённых и незамкнутых телах. Сеть не используется.
