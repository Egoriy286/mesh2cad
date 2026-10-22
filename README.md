# STL → LLM

Минимальное приложение: импорт STL, просмотр и прозрачный pipeline подготовки данных
для LLM. Всё, что уходит в модель, видно на экране; ничего не переписывается по пути.

## Что осталось

```text
app.py                    Streamlit: импорт STL, 3D-просмотр, pipeline
mesh2cad/geometry.py      загрузка STL, нормализация, изображение 512×512
mesh2cad/pipeline.py      точный запрос к OpenRouter, разбор ответа, артефакты
mesh2cad/prompts.py       system prompt и запрос по умолчанию
mesh2cad/dataset.py       воспроизводимая подготовка DeepCAD
mesh2cad/storage.py       атомарная запись JSON
data/deepcad/             локальный датасет (110 STL, manifest)
tests/                    быстрые проверки без сетевых вызовов
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
   - **2 · Изображение для LLM** — ровно **512 × 512** PNG: четыре ортографических
     вида в сетке 2×2 с подписями front / side / top / isometric. Размер, SHA-256
     и готовая картинка видны и скачиваются.
   - **3 · Ввод** — редактируемые system prompt и запрос пользователя, модель,
     reasoning effort. Ниже показан точный JSON запроса; base64 сокращён только
     в предпросмотре, в скачиваемом `request.json` он полный.
   - **4 · Ответ** — кнопка отправки в OpenRouter, сырой ответ, usage и извлечённый
     CadQuery-код.

Каждый успешный вызов сохраняется в `runs/ui/<время>-<id>/`:

```text
source.stl
llm-input-512.png
request.json      # точный запрос, включая base64
response.json     # сырой ответ
code.py
meta.json         # SHA-256, модель, usage, время
```

## Ключ OpenRouter

Ключ читается из `OPENROUTER_API_KEY` или из локального `.env`:

```sh
OPENROUTER_API_KEY=...
```

`.env` добавлен в `.gitignore`. В интерфейсе ключ скрыт, хранится только в памяти
сессии и не записывается в `runs/`.

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

Тесты проверяют нормализацию, точный размер PNG 512×512, структуру запроса,
извлечение кода и сохранение артефактов. Сеть не используется.
