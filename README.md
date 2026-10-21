# Mesh2CAD: генерация с автоматическим исправлением

Небольшой исследовательский CLI для macOS / Apple Silicon. Получает STL, показывает
модели четыре вида, исполняет предложенный CadQuery-код и даёт до двух исправлений
по traceback либо сравнению target/candidate. Сохраняет Python, STEP/STL и парный
отчёт: первая генерация против лучшего результата harness.

Исходная цель проекта — улучшить mesh2cad на недорогих моделях и со временем
проверить перенос на разные домены и снижение API-стоимости в 5–10 раз.
Текущий прототип проверяет только пользу исправлений на DeepCAD. Польза feedback
не отделена от пользы дополнительных попыток: независимый best-of-three baseline
остаётся следующим экспериментом.

## Рабочая среда в браузере

Интерфейс находится в `frontend/` (Streamlit и локальный Three.js), HTTP API и
очередь задач — в `api/` (FastAPI). Для установки и запуска обоих сервисов:

```sh
UV_PROJECT_ENVIRONMENT=.venv-mesh2cad uv sync --frozen --extra dev --extra workbench
.venv-mesh2cad/bin/python workbench.py
```

Откройте [рабочую среду](http://127.0.0.1:8501). API доступен на
`http://127.0.0.1:8765`; Ctrl-C останавливает оба сервиса. Порты и каталог данных
можно изменить через `--ui-port`, `--api-port` и `--data-dir`.

Начните с офлайн-примера: он действительно исполняет CadQuery и вычисляет метрики,
но использует заранее заданный код и не обращается к модели. Для реконструкции
своего mesh выберите OpenRouter в настройках и укажите ключ; поддерживается
`openai/gpt-6-luna`. Ключ хранится только в памяти API и после перезапуска вводится
заново. Лимит расходов по умолчанию — $0.50; неизвестная стоимость остаётся
зарезервированной, прерванный запрос автоматически не повторяется.

Импортируются STL, OBJ, PLY, OFF и GLB с треугольной поверхностью, до 25 MiB.
STEP пока доступен для экспорта результата. В центре можно вращать исходный mesh
и полученную CAD-модель, включать наложение и каркас; справа — запросить объяснение
или создать исправленную версию. Демо-режим не интерпретирует произвольные правки.
Есть ручное редактирование CadQuery, история версий, выбор по IoU или Chamfer,
повторная оценка и экспорт STEP/STL/Python/PNG/JSON/CSV.

Оценка включает IoU, Chamfer, precision/recall/F-score, H95, ошибки объёма и
bounding box. Параметры и причины недоступности метрик видны в интерфейсе.
Геометрия оценивается и экспортируется в нормализованных единицах исходного mesh;
предсказание независимо не выравнивается. Обратное преобразование сохраняется в
данных импорта. Большой mesh может отображаться упрощённо, оценка использует оригинал.
Задачи и версии хранятся в `.cache/workbench`; долгие операции не блокируют интерфейс.

## Установка

Нужны macOS, Python 3.12 и [uv](https://docs.astral.sh/uv/). Для CLI-экспериментов
также нужен установленный Codex CLI с авторизацией ChatGPT Plus (`codex login`). Подписка включает Codex, но имеет свои
лимиты; токены в отчёте не являются измеренной стоимостью API.
См. [условия доступа](https://learn.chatgpt.com/docs/pricing) и
[запуск Codex из скриптов](https://learn.chatgpt.com/docs/non-interactive-mode).

```sh
export UV_PROJECT_ENVIRONMENT=.venv-mesh2cad
uv sync --frozen --extra dev
.venv-mesh2cad/bin/python main.py --help
```

Отдельное окружение сохраняет существующую `.venv`. Все зависимости зафиксированы
в `uv.lock`; GPU, CUDA, веса CADENA/cadrille и окружение DeepCAD не требуются.
CadQuery устанавливается нативно; см. [документацию](https://cadquery.readthedocs.io/en/latest/installation.html).

## Запуск

Один запрос проверяет доступность выбранной модели, протокол, macOS sandbox и
экспорт на простой фигуре:

```sh
.venv-mesh2cad/bin/python main.py run --smoke-only --output runs/smoke
```

Модель по умолчанию — `gpt-6-luna`, reasoning effort — `medium`. При недоступности
модели запуск завершается с диагностикой; автоматической замены нет. Модель можно
выбрать явно через `--model`.

Подготовка [DeepCAD test meshes](https://huggingface.co/datasets/maksimko123/deepcad_test_mesh/tree/main):

```sh
.venv-mesh2cad/bin/python main.py prepare --output data/deepcad
```

Скрипт фиксирует SHA ревизии Hugging Face, перемешивает отсортированные имена с
seed `42` и скачивает только выбранные STL: 10 для разработки и 100 отдельных для
оценки. Сохраняет хэши, исходные mesh, нормализованные STL, обратимые преобразования
и кеш изображений. Ошибочная деталь остаётся в manifest с явным статусом.
Повторная подготовка продолжает тот же manifest и проверяет целостность кеша.

Для локальных данных:

```sh
.venv-mesh2cad/bin/python main.py prepare --source-dir /path/to/stls \
  --output data/local --dev-count 2 --test-count 5
```

Сначала настройка на development subset, затем отдельный evaluation run:

```sh
.venv-mesh2cad/bin/python main.py run --split dev --limit 10 --output runs/dev
.venv-mesh2cad/bin/python main.py run --split test --limit 50 --output runs/test
```

Каждый новый run автоматически делает один preflight-запрос на простой фигуре.
Сравниваемые первая генерация и harness используют один и тот же первый кандидат;
повторно генерировать baseline не нужно. Все запросы последовательные.

Продолжение после прерывания/лимита Plus и расширение оценки до 100 деталей:

```sh
.venv-mesh2cad/bin/python main.py run --split test --limit 50 --output runs/test --resume
.venv-mesh2cad/bin/python main.py run --split test --limit 100 --output runs/test --resume
.venv-mesh2cad/bin/python main.py report runs/test
```

Коды завершения: `0` — закончено; `2` — лимит модели; `130` — Ctrl-C; `1` — другая
ошибка. Возобновление сохраняет завершённые попытки и уже полученный код. После
исправления инфраструктурной ошибки незавершённый запрос можно повторить; все
запуски, включая незавершённые и quota-retries, остаются в учёте.

`--max-attempts` принимает 1–3, `--execution-timeout` по умолчанию 30 секунд,
`--generation-timeout` — 180 секунд. Дополнительные параметры: `--manifest`,
`--n-points` (8192), `--seed`, `--reasoning-effort`.

## Протокол и результаты

- Вход центрируется по bounding box, длиннейшая сторона становится равной 1.
  Предсказание не центрируется и не масштабируется независимо. Экспорт использует
  эти нормализованные единицы; обратная матрица хранится в manifest и `input.json`.
- Модель возвращает JSON с единственным полем `code`. Программа должна создать
  непустой валидный `cq.Workplane` в переменной `r`; экспорт делает wrapper.
- Каждое обращение — новая ephemeral-сессия Codex в пустом временном каталоге.
  Пользовательская конфигурация, shell, web, MCP, plugins и подагенты отключены;
  неожиданные tool-события прерывают запрос и дают `protocol_violation`.
- CadQuery выполняется в отдельном процессе под macOS Seatbelt: доступны библиотеки
  Python и каталог одной попытки, сеть и чтение dataset/репозитория/секретов закрыты.
  Без работающего sandbox исполнение прекращается; небезопасного fallback нет.
- Лучший валидный кандидат выбирается по минимальному symmetric squared Chamfer
  distance (сумма двух средних квадратов расстояний, без множителя). По 8192 точкам
  поверхности с фиксированным seed; масштаб и положение учитываются.
- IoU вычисляется объёмными boolean-операциями через manifold3d для замкнутых
  корректных тел. Неприменимость и ошибки метрики имеют отдельные статусы.

```text
runs/test/
  config.json, dataset.json, selection.json, prompt.txt, state.json
  preflight/invocation-000/              # отдельный учёт startup-проверки
  items/<id>/
    input.json, target.png, result.json
    baseline.py, baseline.stl, baseline.step  # mesh/STEP только при успехе
    best.py, best.stl, best.step
    attempts/000/
      code.py, comparison.png
      generation-000/                    # prompt, schema, JSONL, stderr, usage
      execution/                         # код, STEP/STL, sandbox, traceback, logs
  results.csv, summary.json, report.md
```

Отчёт учитывает все выбранные ID, в том числе pending и ошибки. Показывает CAD
validity, парное изменение Chamfer на общих успешных примерах, доступность IoU,
baseline/best/last, вызовы, известные токены и время. Неизвестный usage остаётся
неизвестным. Расходы preflight вынесены отдельно. Для одного кандидата используются
его собственные CD и IoU; выбор разных «лучших» кандидатов для разных метрик исключён.

Промпты и параметры нужно завершить на dev subset до оценки test. Конфигурация run
фиксирует код harness, промпты, версии зависимостей, модель и входной manifest.
`--resume` отвергает изменения этих параметров; разрешено лишь расширить исходный
префикс выборки. Для новой настройки создайте новый каталог run.

## Проверки

```sh
.venv-mesh2cad/bin/python -m pytest
.venv-mesh2cad/bin/ruff check mesh2cad tests main.py
```

Тесты не обращаются к моделям. Они проверяют геометрию, настоящие CadQuery-экспорты,
ошибки/таймауты, аудит fake Codex CLI, quota/resume и знаменатели отчётов.
Под внешним sandbox тесты самого macOS sandbox могут быть пропущены: для реальной
проверки запустите CLI smoke из обычного терминала.
