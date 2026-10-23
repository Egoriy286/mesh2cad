"""Compare shared-first-response baselines and native harness runs without API calls."""

import csv
import io
import json
from pathlib import Path

import numpy as np

from mesh2cad.harness import TERMINAL_ITEMS

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/benchmarks/20261023-sol-hard50-interior50"
SOURCES = [("Luna", ROOT / "output/benchmarks/20261023-hard50-interior50", 5), ("Sol", OUT, 2)]


def read(path):
    return json.loads(path.read_text())


def main():
    rows = []
    for model, folder, cycles in SOURCES:
        protocol = read(folder / "protocol.json")
        for batch in protocol["batches"]:
            run = Path(batch["run"])
            state = read(run / "run.json")
            for item in state["items"]:
                attempts = item["attempts"]
                finished = item["status"] in TERMINAL_ITEMS
                for first, mode in [
                    (True, f"{model}: 1 промпт"),
                    (False, f"{model}: harness ≤{cycles}"),
                ]:
                    chosen = attempts[:1] if first else attempts
                    scores = [(a.get("iou") or {}).get("iou") for a in chosen]
                    scored = [s for s in scores if s is not None]
                    calls = [
                        c for a in chosen for c in a.get("calls", []) if c["status"] == "received"
                    ]
                    rows.append(
                        dict(
                            mode=mode,
                            model=protocol["model"],
                            domain=batch["domain"],
                            id=item["id"],
                            name=item.get("name"),
                            batch=batch["batch"],
                            finished=finished,
                            iou=max(scored) if scored else None,
                            cad_valid=any(
                                a.get("execution", {}).get("status") == "valid" for a in chosen
                            ),
                            calls=len(calls),
                            tokens=sum(
                                (c.get("usage") or {}).get("total_tokens", 0) for c in calls
                            ),
                            cost_usd=sum((c.get("usage") or {}).get("cost", 0) or 0 for c in calls),
                            input_sha256=item["input_sha256"],
                            source=str(run / "run.json"),
                        )
                    )
    controls = []
    for domain in ["deepcad", "interior300", "all"]:
        for mode in dict.fromkeys(r["mode"] for r in rows):
            selected = [
                r for r in rows if r["mode"] == mode and (domain == "all" or r["domain"] == domain)
            ]
            done = [r for r in selected if r["finished"]]
            scores = [r["iou"] or 0 for r in done]
            controls.append(
                dict(
                    domain=domain,
                    mode=mode,
                    selected=len(selected),
                    finished=len(done),
                    mean_iou=float(np.mean(scores)) if done else None,
                    cad_valid=sum(r["cad_valid"] for r in done),
                    iou_ge_0_9=sum(s >= 0.9 for s in scores),
                    calls=sum(r["calls"] for r in selected),
                    tokens=sum(r["tokens"] for r in selected),
                    cost_usd=sum(r["cost_usd"] for r in selected),
                )
            )
    data = dict(
        rows=rows,
        controls=controls,
        notes=[
            "Одинаковые 100 STL и SHA-256; неудача завершённой реконструкции = 0.",
            "Один промпт — первая итерация того же запуска. Расходы режимов перекрываются: их нельзя суммировать.",
            "Luna: до 5 итераций; Sol: до 2 итераций всего, включая первую; остановка при IoU ≥ 0.9.",
            "Токены и USD относятся к сохранённым ответам. Сверка фактического списания — audit.json каждого запуска.",
        ],
    )
    (OUT / "comparison.json").write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=list(controls[0]))
    writer.writeheader()
    writer.writerows(controls)
    (OUT / "comparison.csv").write_text(stream.getvalue())
    lines = [
        "# Сравнение Luna и Sol на одинаковых 100 STL",
        "",
        *data["notes"],
        "",
        "| Домен | Режим | Завершено | Средний IoU | CAD валиден | IoU ≥ 0.9 | USD сохранённых ответов |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for c in controls:
        score = "—" if c["mean_iou"] is None else f"{c['mean_iou']:.4f}"
        lines.append(
            f"| {c['domain']} | {c['mode']} | {c['finished']}/{c['selected']} | {score} | {c['cad_valid']} | {c['iou_ge_0_9']} | {c['cost_usd']:.6f} |"
        )
    (OUT / "comparison.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(controls, ensure_ascii=False))


if __name__ == "__main__":
    main()
