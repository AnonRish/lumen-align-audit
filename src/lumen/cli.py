"""Command-line interface. After `pip install -e .`, available as `lumen`."""
from __future__ import annotations
import argparse
import sys
import os


def cmd_train_organism(args):
    from .neuralese.toy_transformer import train_toy_transformer
    print(f"Training reference organism ({args.steps} steps, seed={args.seed})...")
    result = train_toy_transformer(steps=args.steps, seed=args.seed, verbose=True, log_every=max(args.steps // 6, 1))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    result.model.save(args.out)
    print(f"\nFinal EVAL accuracy:   {result.final_eval_acc:.1%}")
    print(f"Final DEPLOY accuracy: {result.final_deploy_acc:.1%}")
    print(f"Saved to {args.out}")


def cmd_audit_organism(args):
    from .audit.pipeline import run_toy_organism_audit
    from .audit.report import save_report
    from .model_organisms.toy_backdoor import load_reference_organism
    print("Loading reference organism and running full audit "
          "(neuralese decoding + activation probes + activation patching)...")
    model = load_reference_organism(args.checkpoint)
    report = run_toy_organism_audit(model, n_patch_pairs=args.patch_pairs)
    print()
    print(report.to_markdown())
    path = save_report(report, args.reports_dir)
    print(f"\nSaved to {path}")


def cmd_audit_llm(args):
    from .llm_client import make_respond_fn
    from .audit.pipeline import run_llm_audit
    from .audit.report import save_report
    respond_fn = make_respond_fn(model=args.model)
    print(f"Running persona '{args.persona}' against {args.model}...")
    report = run_llm_audit(respond_fn, args.persona, args.task, bias_hint_option=args.bias_hint)
    print()
    print(report.to_markdown())
    path = save_report(report, args.reports_dir)
    print(f"\nSaved to {path}")


def cmd_list_personas(args):
    from .model_organisms.personas import PERSONAS
    for p in PERSONAS:
        print(f"{p.name:28s} [{p.category}]")
        print(f"  {p.description}")
        print(f"  citation: {p.citation}")
        print()


def cmd_dashboard(args):
    try:
        from dashboard.app import create_app
    except ImportError:
        print("Dashboard requires the repo root on PYTHONPATH. Run this from the repo root, "
              "or `cd dashboard && python app.py` directly.", file=sys.stderr)
        sys.exit(1)
    app = create_app(reports_dir=args.reports_dir)
    print(f"Dashboard running at http://127.0.0.1:{args.port}")
    app.run(port=args.port, debug=False)


def build_parser():
    parser = argparse.ArgumentParser(prog="lumen", description=(
        "Lumen -- Technical Alignment & Interpretability Audit Framework "
        "(AI 2040 Plan A: faithful CoT, neuralese decoding, deception probes, model organisms)."
    ))
    sub = parser.add_subparsers(dest="command", required=True)

    p1 = sub.add_parser("train-organism", help="Train the reference toy model organism from scratch.")
    p1.add_argument("--steps", type=int, default=3000)
    p1.add_argument("--seed", type=int, default=0)
    p1.add_argument("--out", default="checkpoints/reference_organism.npz")
    p1.set_defaults(func=cmd_train_organism)

    p2 = sub.add_parser("audit-organism", help="Run the full audit pipeline against the toy organism.")
    p2.add_argument("--checkpoint", default="checkpoints/reference_organism.npz")
    p2.add_argument("--patch-pairs", type=int, default=100, dest="patch_pairs")
    p2.add_argument("--reports-dir", default="audit_reports", dest="reports_dir")
    p2.set_defaults(func=cmd_audit_organism)

    p3 = sub.add_parser("audit-llm", help="Run a model-organism persona + faithful-CoT audit against a live Claude model. Needs ANTHROPIC_API_KEY.")
    p3.add_argument("--persona", required=True, help="See `lumen list-personas`.")
    p3.add_argument("--task", required=True, help="The benign task to embed the persona's scenario around.")
    p3.add_argument("--model", default="claude-sonnet-5")
    p3.add_argument("--bias-hint", default=None, dest="bias_hint",
                     help="Optional: an answer option (e.g. '(A)') to test bias injection with.")
    p3.add_argument("--reports-dir", default="audit_reports", dest="reports_dir")
    p3.set_defaults(func=cmd_audit_llm)

    p4 = sub.add_parser("list-personas", help="List available model-organism personas.")
    p4.set_defaults(func=cmd_list_personas)

    p5 = sub.add_parser("dashboard", help="Launch the local Flask dashboard over saved audit reports.")
    p5.add_argument("--port", type=int, default=5050)
    p5.add_argument("--reports-dir", default="audit_reports", dest="reports_dir")
    p5.set_defaults(func=cmd_dashboard)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
