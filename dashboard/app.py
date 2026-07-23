"""
Local Flask dashboard over saved audit reports. Read-mostly: the only
write action is "run a new toy-organism audit," which is fully local
(trains/loads the toy transformer, no network calls). LLM audits are
triggered via the CLI (`lumen audit-llm`, needs your own API key) and just
show up here once saved.

Run with `lumen dashboard` (after `pip install -e .`) or directly:
    cd dashboard && python app.py
"""
from __future__ import annotations
import time
import os
import sys
from flask import Flask, render_template, redirect, url_for, abort

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from lumen.audit.report import list_reports, load_report, save_report, DEFAULT_REPORTS_DIR  # noqa: E402
from lumen.audit.pipeline import run_toy_organism_audit  # noqa: E402

PILLARS = [
    {"ch": "CH.1", "name": "Faithful CoT", "status": "tested",
     "note": "perturbation + bias-injection tests, hand-written fixtures + live API"},
    {"ch": "CH.2", "name": "Neuralese Decoding", "status": "tested",
     "note": "logit lens + tuned lens on the trained reference organism"},
    {"ch": "CH.3", "name": "Deception Probes", "status": "tested",
     "note": "activation probes (organism) + behavioral probe (API models)"},
    {"ch": "CH.4", "name": "Model Organisms", "status": "tested",
     "note": "toy trigger-conditioned organism + prompted persona library"},
]


def _svg_trace(values, labels, width=480, height=120, color_var="--amber", y_label=""):
    """Server-side oscilloscope-style polyline. values: list[float] (already
    the metric to plot, e.g. mean rank or flip rate). Returns raw SVG string."""
    if not values:
        return ""
    pad_l, pad_r, pad_t, pad_b = 34, 14, 14, 24
    plot_w, plot_h = width - pad_l - pad_r, height - pad_t - pad_b
    vmin, vmax = min(values), max(values)
    if vmax - vmin < 1e-9:
        vmax = vmin + 1.0
    n = len(values)
    xs = [pad_l + (i / max(n - 1, 1)) * plot_w for i in range(n)]
    ys = [pad_t + plot_h - ((v - vmin) / (vmax - vmin)) * plot_h for v in values]
    points = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys))
    dots = "".join(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3" fill="var({color_var})"/>' for x, y in zip(xs, ys))
    gridlines = "".join(
        f'<line x1="{pad_l}" y1="{pad_t + plot_h*f:.1f}" x2="{width-pad_r}" y2="{pad_t + plot_h*f:.1f}" '
        f'stroke="var(--border)" stroke-width="1"/>' for f in (0, 0.5, 1.0)
    )
    tick_labels = "".join(
        f'<text x="{x:.1f}" y="{height-6}" font-family="var(--font-mono)" font-size="9.5" '
        f'fill="var(--text-dim)" text-anchor="middle">{lab}</text>' for x, lab in zip(xs, labels)
    )
    return (
        f'<svg viewBox="0 0 {width} {height}" width="100%" height="{height}" role="img" aria-label="{y_label} trace">'
        f'{gridlines}'
        f'<polyline points="{points}" fill="none" stroke="var({color_var})" stroke-width="2" '
        f'style="filter: drop-shadow(0 0 4px rgba(255,180,84,0.5))"/>'
        f'{dots}{tick_labels}'
        f'<text x="{pad_l}" y="12" font-family="var(--font-mono)" font-size="10" fill="var(--text-muted)">{y_label}</text>'
        f'</svg>'
    )


def _svg_bars(labels, values, width=480, height=120, color_var="--cyan", y_label=""):
    if not values:
        return ""
    pad_l, pad_r, pad_t, pad_b = 34, 14, 14, 24
    plot_w, plot_h = width - pad_l - pad_r, height - pad_t - pad_b
    n = len(values)
    bar_w = plot_w / n * 0.55
    gap = plot_w / n
    bars, labs = [], []
    for i, (lab, v) in enumerate(zip(labels, values)):
        x = pad_l + i * gap + (gap - bar_w) / 2
        bar_h = max(v, 0.02) * plot_h
        y = pad_t + plot_h - bar_h
        bars.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{bar_h:.1f}" '
                    f'rx="2" fill="var({color_var})" opacity="0.85"/>')
        labs.append(f'<text x="{x+bar_w/2:.1f}" y="{height-6}" font-family="var(--font-mono)" font-size="9.5" '
                    f'fill="var(--text-dim)" text-anchor="middle">{lab}</text>')
    return (
        f'<svg viewBox="0 0 {width} {height}" width="100%" height="{height}" role="img" aria-label="{y_label} bars">'
        f'<line x1="{pad_l}" y1="{pad_t+plot_h}" x2="{width-pad_r}" y2="{pad_t+plot_h}" stroke="var(--border)" stroke-width="1"/>'
        + "".join(bars) + "".join(labs) +
        f'<text x="{pad_l}" y="12" font-family="var(--font-mono)" font-size="10" fill="var(--text-muted)">{y_label}</text>'
        f'</svg>'
    )


def create_app(reports_dir: str = DEFAULT_REPORTS_DIR) -> Flask:
    app = Flask(__name__)
    app.jinja_env.filters["timestamp_to_str"] = lambda ts: time.strftime("%Y-%m-%d %H:%M", time.localtime(ts))

    @app.route("/")
    def index():
        reports = list_reports(reports_dir)
        return render_template("index.html", reports=reports, pillars=PILLARS)

    @app.route("/run/organism", methods=["POST"])
    def run_organism():
        report = run_toy_organism_audit()
        path = save_report(report, reports_dir)
        return redirect(url_for("view_report", filename=os.path.basename(path)))

    @app.route("/report/<path:filename>")
    def view_report(filename):
        path = os.path.join(reports_dir, filename)
        if not os.path.isfile(path):
            abort(404)
        payload = load_report(path)
        data = payload["data"]
        traces = {}
        if payload["kind"] == "toy_organism":
            lens = data.get("logit_lens_summary", {})
            traces["logit_lens"] = _svg_trace(list(lens.values()), [k.replace("after_block_", "L").replace("embeddings", "emb") for k in lens.keys()],
                                              y_label="mean rank of true answer (lower = more recoverable)")
            patch = data.get("patching_summary", {})
            traces["patching"] = _svg_bars(list(patch.keys()), list(patch.values()), color_var="--cyan",
                                           y_label="max flip rate when patched")
        return render_template("report.html", payload=payload, data=data, traces=traces, filename=filename)

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(port=5050, debug=True)
