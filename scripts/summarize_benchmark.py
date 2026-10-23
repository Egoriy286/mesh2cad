"""Create portable tables and editable Beamer figures from measured benchmark rows."""

from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "output/benchmarks/20261023"
DECK = ROOT / "output/latex/mesh2cad-research"
CONDITIONS = ("cheap_one_prompt", "cheap_harness", "expensive_one_prompt")
LABELS = ("Дешёвая, один промпт", "Дешёвая + harness", "Дорогая, один промпт")


def read(path):
    return json.loads(path.read_text())


def main():
    summary = read(BENCH / "summary.json")
    groups = summary["groups"]
    rows = list(csv.DictReader((BENCH / "items.csv").open()))
    protocol = read(BENCH / "protocol.json")
    assert summary["budget"]["unknown_cost_calls"] == 0
    for domain, group in groups.items():
        ids = [
            {r["id"] for r in rows if r["domain"] == domain and r["condition"] == c}
            for c in CONDITIONS
        ]
        assert ids[0] == ids[1] == ids[2]
        assert len({group[c]["n"] for c in CONDITIONS}) == 1

    spend = sum(
        groups[d][c]["cost_usd"] for d in groups for c in ("cheap_harness", "expensive_one_prompt")
    )
    lines = [
        "# Mesh2CAD: сравнение моделей и harness",
        "",
        "Дата эксперимента: 23 октября 2026, Europe/Moscow.",
        "",
        "Модели: `openai/gpt-6-luna` и `openai/gpt-6-sol`. "
        "Основной метод: classic harness, до пяти генераций. "
        "У baseline один вызов генерации, без предварительного анализа.",
        "",
        "[Полное описание метода с визуализациями](../../../docs/method.md)",
        "",
        "## Измеренные результаты",
        "",
        "Средний IoU включает неудачные реконструкции как 0. "
        "В колонке «пригодный объём» показана доступность IoU, "
        "в колонке «CAD исполнился» - успешное исполнение хотя бы одной программы.",
        "",
        "| Домен | Условие | N | Mean IoU | Медиана | CAD исполнился | Пригодный объём | IoU ≥ 0.9 | USD всего | USD/объект | Вызовы | Секунд/объект |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for domain in groups:
        for condition, label in zip(CONDITIONS, LABELS):
            g = groups[domain][condition]
            lines.append(
                f"| {domain} | {label} | {g['n']} | {g['mean_iou']:.4f} | "
                f"{g['median_iou']:.4f} | {g['cad_valid']}/{g['n']} | "
                f"{g['scorable']}/{g['n']} | {g['success_0_9']}/{g['n']} | "
                f"{g['cost_usd']:.6f} | {g['cost_usd'] / g['n']:.6f} | "
                f"{g['calls']} | {g['mean_seconds']:.1f} |"
            )
    lines += [
        "",
        "Время включает подготовку входа, вызовы, исполнение, IoU и диагностические рендеры. "
        "Задачи выполнялись параллельно, поэтому это время в условиях данного запуска. "
        "USD отражает только стоимость API, без цены локальных вычислений.",
        "",
        "Ошибки исполнения по лимиту памяти учитываются в результатах. "
        "Для одной генерации это неудачный выход; harness может исправить конструкцию "
        "по сообщению watchdog. Лимит 2048 МиБ одинаков, оплаченный ответ не запрашивается заново. "
        "Счётчики memory_failures сохранены в items.csv и summary.json.",
        "",
        "## Парные разности IoU",
        "",
        "Bootstrap по объектам, 20 000 повторов, seed 42, percentile 95% CI.",
        "",
    ]
    for domain in groups:
        for baseline in ("cheap_one_prompt", "expensive_one_prompt"):
            g = groups[domain]["harness_minus_" + baseline]
            lines.append(
                f"- {domain}, harness минус {baseline}: **{g['mean']:+.4f}**, "
                f"95% CI [{g['ci95'][0]:+.4f}; {g['ci95'][1]:+.4f}], "
                f"улучшение у {g['improved']}/{g['n']} объектов."
            )
    lines += [
        "",
        "## Ресурсы и интерпретация",
        "",
        f"Фактически оплаченные завершённые реконструкции: **${spend:.6f}**. "
        f"Полный журнал новых расходов: **${summary['budget']['spent']:.6f}**, "
        f"заданный предел ${summary['budget']['cap']:.2f}. "
        "Первый ответ дешёвой модели разделяется между baseline и harness, "
        "поэтому стоимость этих двух строк не суммируется при расчёте фактического расхода.",
        "",
        "План: 50 деталей + 50 объектов. Фактическая когорта - префикс заранее "
        "зафиксированного списка с ограничением по ресурсам. Interior300 чередует "
        "4 IndoorCAD и 1 Poly Haven. Во всех условиях внутри домена одни и те же ID и SHA-256. "
        "Незавершённые объекты не входят в основную таблицу и перечислены в summary.json. "
        "Незапущенные объекты не считаются неудачами.",
        "",
        f"Незавершённых объектов в журнале: {len(summary['incomplete'])}.",
        "",
        "Дорогая модель с harness является дополнительным условием и не включена "
        "в этот запуск: доступный бюджет направлен на три обязательных условия.",
        "",
        "Это исследовательская локальная оценка. Список DeepCAD сохраняет исторический "
        "порядок выбранных test-объектов, поэтому он не является новым нетронутым holdout. "
        "У Interior300 нет официального split, выборка ограничена корректными объёмами. "
        "Базовый промпт classic использует формулировку mechanical part в обоих доменах. "
        "Этот перенос промпта также может влиять на результат интерьеров. "
        "Ни один объект не исключается за низкий полученный IoU.",
        "",
        "Одной генерации baseline на объект недостаточно, чтобы оценить "
        "вариативность повторных ответов. Дополнительный контроль независимых "
        "генераций с тем же числом вызовов нужен для выделения вклада именно "
        "обратной связи из общего эффекта дополнительного вычисления.",
        "",
        "## Воспроизведение",
        "",
        "```sh",
        ".venv/bin/python -m mesh2cad.benchmark --run --limit 50 --budget 1.35",
        ".venv/bin/python -m mesh2cad.benchmark",
        ".venv/bin/python scripts/summarize_benchmark.py",
        "```",
        "",
        "Без `--run` оплаченные запросы не выполняются. Завершённые реконструкции "
        "повторно не генерируются. Повышение бюджета требует явного изменения "
        "budget.json, аргумент CLI не меняет уже сохранённый лимит.",
        "",
        "- [Пообъектные метрики](items.csv)",
        "- [Агрегированные метрики и доверительные интервалы](summary.json)",
        "- [Протокол, списки входов и хеши реализации](protocol.json)",
        "- [Бюджет и зафиксированные тарифы OpenRouter](budget.json)",
        "",
        "- [Независимая проверка запросов, оценок и оплаты](audit.json)",
        "",
        "Полные запросы, ответы, программы, STL/STEP и checkpoints лежат локально "
        "в `runs/` этой папки. Пути к ним указаны в items.csv; ключи API туда не записываются.",
    ]
    (BENCH / "README.md").write_text("\n".join(lines) + "\n")

    tex = ["% Generated from output/benchmarks/20261023/summary.json and items.csv."]
    for domain, title in (
        ("deepcad", "DeepCAD: механические детали"),
        ("interior300", "Interior300: предметы интерьера"),
    ):
        n = groups[domain]["cheap_harness"]["n"]
        tex += [
            r"\begin{frame}{" + title + r"}\relax",
            r"\textbf{N = " + str(n) + r", одинаковые входы}",
            r"\par\medskip",
            r"{\smalltext",
            r"\begin{tabular}{@{}L{.33\textwidth}rrrr@{}}",
            r"\toprule",
            r"\textbf{Постановка} & \textbf{Mean IoU} & \textbf{Объём} & \textbf{IoU $\geq.9$} & \textbf{USD/объект}\\",
            r"\midrule",
        ]
        for condition, label in zip(CONDITIONS, LABELS):
            g = groups[domain][condition]
            tex.append(
                f"{label} & {g['mean_iou']:.4f} & {g['scorable']}/{n} & "
                f"{g['success_0_9']}/{n} & {g['cost_usd'] / n:.6f} " + r"\\"
            )
        delta = groups[domain]["harness_minus_cheap_one_prompt"]
        expensive = groups[domain]["harness_minus_expensive_one_prompt"]
        tex += [
            r"\bottomrule",
            r"\end{tabular}}",
            r"\par\bigskip",
            r"\textbf{$\Delta$IoU к первому ответу Luna: " + f"{delta['mean']:+.4f}" + r"}",
            r"\par\smallskip",
            "95\\% CI ["
            + f"{delta['ci95'][0]:+.4f}; {delta['ci95'][1]:+.4f}"
            + "]. "
            + f"Улучшены {delta['improved']}/{n}.",
            r"\par\medskip{\smalltext",
            r"$\Delta$IoU к Sol: "
            + f"{expensive['mean']:+.4f}"
            + ". "
            + "95\\% CI ["
            + f"{expensive['ci95'][0]:+.4f}; {expensive['ci95'][1]:+.4f}"
            + "].}",
            r"\source{Парный bootstrap, 95\% CI. Неудачи: IoU = 0. "
            r"Цена включает все попытки.}",
            r"\end{frame}",
        ]
        target = DECK / "images" / ("benchmark-" + domain + ".dat")
        by_id = {
            c: {r["id"]: r for r in rows if r["domain"] == domain and r["condition"] == c}
            for c in CONDITIONS
        }
        data = ["first best expensive"]
        for item_id in sorted(by_id["cheap_harness"]):
            data.append(" ".join(by_id[c][item_id]["iou_failure_zero"] for c in CONDITIONS))
        target.write_text("\n".join(data) + "\n")

    tex += [
        r"\begin{frame}{Эффект по отдельным объектам}\relax",
        r"\begin{columns}[T,onlytextwidth]",
    ]
    for domain, title in (("deepcad", "DeepCAD"), ("interior300", "Interior300")):
        tex += [
            r"\begin{column}{.48\textwidth}",
            r"\centering\textbf{" + title + "}",
            r"\begin{tikzpicture}\begin{axis}[width=\linewidth,height=49mm,",
            r"xmin=0,xmax=1.03,ymin=0,ymax=1.03,axis lines=left,",
            r"xlabel={IoU одного промпта},ylabel={IoU с harness},",
            r"label style={font=\scriptsize},tick label style={font=\scriptsize},",
            r"xtick={0,.5,1},ytick={0,.5,1},grid=major,grid style={gray!15}]",
            r"\addplot[gray,dashed,domain=0:1,samples=2]{x};",
            r"\addplot[only marks,mark=*,mark size=2,Blue,opacity=.75]",
            r"table[x=first,y=best]{images/benchmark-" + domain + r".dat};",
            r"\end{axis}\end{tikzpicture}\end{column}",
        ]
    tex += [
        r"\end{columns}",
        r"\source{Точка: один объект. Выше диагонали: улучшение. "
        r"Совпадающие точки накладываются.}",
        r"\end{frame}",
    ]

    tex += [
        r"\begin{frame}{Ресурсы и границы сравнения}\relax",
        r"\begin{itemize}",
        r"\item Обработано 15 + 15 объектов из плана 50 + 50.",
        r"\par\medskip",
        r"\item Первый ответ Luna общий для baseline и harness.",
        r"\par\medskip",
        r"\item Полный расход API: \textbf{\$" + f"{summary['budget']['spent']:.4f}" + "}.",
        r"\par\medskip",
        r"\item Повторные ответы модели пока не оценены.",
        r"\end{itemize}",
        r"\source{Локальные подвыборки, не новый holdout. "
        r"Промпт mechanical part в обоих доменах. Лимит памяти 2048 МиБ.}",
        r"\end{frame}",
        r"\begin{frame}{Выводы из результатов}\relax",
        r"\begin{itemize}",
    ]
    for domain, title in (("deepcad", "DeepCAD"), ("interior300", "Interior300")):
        g = groups[domain]
        tex += [
            r"\item "
            + title
            + ": IoU "
            + f"{g['cheap_harness']['mean_iou']:.4f}"
            + " против "
            + f"{g['expensive_one_prompt']['mean_iou']:.4f}.",
            "API дешевле Sol в "
            + f"{g['expensive_one_prompt']['cost_usd'] / g['cheap_harness']['cost_usd']:.2f}"
            + " раза.",
            r"\par\medskip",
        ]
    tex += [
        r"\item Интерьеры остаются сложными: 0/15 с IoU $\geq0.9$ у harness.",
        r"\end{itemize}",
        r"\source{Качество не доказано эквивалентным. "
        r"Цена API не включает локальные вычисления.}",
        r"\end{frame}",
        r"\begin{frame}{Планы на будущее}\relax",
        r"\begin{enumerate}",
        r"\item Расширить до 50 + 50 и повторить генерации.",
        r"\par\medskip",
        r"\item Сравнить с генерациями без обратной связи при том же числе вызовов.",
        r"\par\medskip",
        r"\item Измерить вклад IoU, карты ошибок и памяти.",
        r"\par\medskip",
        r"\item Добавить метрики поверхности и общий бенчмарк с методами обзора.",
        r"\end{enumerate}",
        r"\source{При дополнительном бюджете: Sol с харнессом.}",
        r"\end{frame}",
    ]
    (DECK / "benchmark-results.tex").write_text("\n".join(tex) + "\n")
    notes = (DECK / "speaker-notes.template.md").read_text()
    for domain, token in (("deepcad", "DEEPCAD"), ("interior300", "INTERIOR300")):
        g = groups[domain]
        result_text = (
            f"Во всех условиях обработано {g['cheap_harness']['n']} одинаковых объектов. "
            f"Mean IoU первого ответа Luna {g['cheap_one_prompt']['mean_iou']:.4f}, "
            f"с харнессом {g['cheap_harness']['mean_iou']:.4f}, "
            f"одного ответа Sol {g['expensive_one_prompt']['mean_iou']:.4f}. "
            f"Цена API на объект для этих условий: "
            f"${g['cheap_one_prompt']['cost_usd'] / g['cheap_one_prompt']['n']:.6f}, "
            f"${g['cheap_harness']['cost_usd'] / g['cheap_harness']['n']:.6f} и "
            f"${g['expensive_one_prompt']['cost_usd'] / g['expensive_one_prompt']['n']:.6f}. "
            f"Харнесс дешевле одного ответа Sol в "
            f"{g['expensive_one_prompt']['cost_usd'] / g['cheap_harness']['cost_usd']:.2f} раза."
        )
        notes = notes.replace("{{" + token + "_RESULTS}}", result_text)
    notes = notes.replace("{{SPEND_USD}}", f"{summary['budget']['spent']:.6f}")
    assert "{{" not in notes, "Unresolved speaker note field"
    (DECK / "speaker-notes.md").write_text(notes)
    print(
        json.dumps(
            {
                "n": {d: groups[d]["cheap_harness"]["n"] for d in groups},
                "spent": summary["budget"]["spent"],
                "status": protocol["status"],
            }
        )
    )


if __name__ == "__main__":
    main()
