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


def cmd_control_eval(args):
    from .control.control_eval import run_control_evaluation
    print("Running control evaluation (monitor + defer-to-trusted protocol sweep)...")
    report = run_control_evaluation(n_train=args.n_train, n_eval=args.n_eval)
    print()
    print(report.to_markdown())


def cmd_verify_wrapper(args):
    from .control.formal_wrapper import run_exhaustive_verification
    from .model_organisms.toy_backdoor import load_reference_organism
    print("Running exhaustive formal verification of the wrapped organism...")
    model = load_reference_organism(args.checkpoint)
    result = run_exhaustive_verification(model)
    print()
    print(result.to_markdown())


def cmd_elicitation_test(args):
    from .elicitation.password_locked import train_password_locked_organism
    from .elicitation.sandbagging_detection import run_sandbagging_diagnosis
    print(f"Training a password-locked organism ({args.steps} steps)...")
    locked = train_password_locked_organism(steps=args.steps, seed=args.seed)
    print(f"unlocked accuracy: {locked.unlocked_accuracy:.1%}  locked accuracy: {locked.locked_accuracy:.1%}")
    print("\nRunning sandbagging-vs-genuine-ceiling diagnosis "
          "(this trains a comparison model + runs several fine-tune trials, ~15-30s)...")
    report = run_sandbagging_diagnosis(locked.model)
    print()
    print(report.to_markdown())


def cmd_full_audit(args):
    from .audit.pipeline import run_full_audit
    from .audit.report import save_report
    print("Running the full audit: all pillars against the reference organism...")
    result = run_full_audit(n_patch_pairs=args.patch_pairs)
    print()
    print(result["assurance_case"].to_markdown())
    path = save_report(result["toy_organism_report"], args.reports_dir)
    print(f"\n(Toy-organism sub-report saved to {path}; the other sub-reports print above but "
          f"aren't separately persisted yet -- see STATUS.md.)")


def cmd_governance_profile(args):
    from .control.autonomy_metrics import profile_toy_organism
    from .governance.agi_profile import classify_toy_organism
    from .model_organisms.toy_backdoor import load_reference_organism
    from .neuralese.toy_transformer import evaluate as eval_fn
    import numpy as np
    model = load_reference_organism(args.checkpoint)
    accs = eval_fn(model, np.random.default_rng(0), n=500)
    task_acc = (accs["EVAL"] + accs["DEPLOY"]) / 2
    classification = classify_toy_organism(profile_toy_organism(), task_accuracy=task_acc)
    print(classification.to_markdown())


def cmd_neuralese_recurrence(args):
    from .neuralese.recurrent_reasoner import train_recurrent_reasoner, NeuraleseRecurrentReasoner, make_recurrent_batch
    from .neuralese.recurrent_decoding import run_stepwise_logit_lens, run_stepwise_patching
    import os
    if os.path.exists(args.checkpoint) and not args.retrain:
        print(f"Loading existing checkpoint from {args.checkpoint} (pass --retrain to force retraining)...")
        model = NeuraleseRecurrentReasoner.load(args.checkpoint)
    else:
        print(f"Training the neuralese recurrent reasoner ({args.steps} steps)...")
        result = train_recurrent_reasoner(steps=args.steps, seed=args.seed, verbose=True, log_every=max(args.steps // 5, 1))
        model = result.model
        os.makedirs(os.path.dirname(args.checkpoint) or ".", exist_ok=True)
        model.save(args.checkpoint)
        print(f"Saved to {args.checkpoint}")

    print("\nRunning stepwise logit lens (can we decode the hidden reasoning trace?)...")
    import numpy as np
    rng = np.random.default_rng(42)
    starts, ops, results, traces = make_recurrent_batch(rng, 300, n_ops=model.cfg.n_ops)
    lens_res = run_stepwise_logit_lens(model, starts, ops, traces)
    for name, rate in zip(lens_res.step_names, lens_res.true_value_top1_rate_by_step):
        print(f"  {name:14s}  true partial sum recoverable: {rate:.1%}")

    print("\nRunning stepwise activation patching (is that hidden state causally load-bearing?)...")
    patch_res = run_stepwise_patching(model, n_pairs=100, seed=0)
    for name, rate in zip(patch_res.step_names, patch_res.flip_rate_by_step):
        print(f"  {name:14s}  donor-trajectory match rate: {rate:.1%}")


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

    p6 = sub.add_parser("control-eval", help="Run the AI Control evaluation (monitor + defer-to-trusted Safety/Usefulness sweep).")
    p6.add_argument("--n-train", type=int, default=1000, dest="n_train")
    p6.add_argument("--n-eval", type=int, default=1000, dest="n_eval")
    p6.set_defaults(func=cmd_control_eval)

    p7 = sub.add_parser("verify-wrapper", help="Exhaustively formally verify the Guaranteed-Safe-AI-style wrapper (checks all 200 possible inputs).")
    p7.add_argument("--checkpoint", default="checkpoints/reference_organism.npz")
    p7.set_defaults(func=cmd_verify_wrapper)

    p8 = sub.add_parser("elicitation-test", help="Train a password-locked organism and run the sandbagging-vs-genuine-ceiling diagnosis.")
    p8.add_argument("--steps", type=int, default=3000)
    p8.add_argument("--seed", type=int, default=0)
    p8.set_defaults(func=cmd_elicitation_test)

    p9 = sub.add_parser("full-audit", help="Run every pillar against the reference organism and produce one Assurance Case.")
    p9.add_argument("--patch-pairs", type=int, default=100, dest="patch_pairs")
    p9.add_argument("--reports-dir", default="audit_reports", dest="reports_dir")
    p9.set_defaults(func=cmd_full_audit)

    p10 = sub.add_parser("governance-profile", help="Classify the reference organism under the A-G-I / Tool-Supervised-Autonomous governance framework.")
    p10.add_argument("--checkpoint", default="checkpoints/reference_organism.npz")
    p10.set_defaults(func=cmd_governance_profile)

    p11 = sub.add_parser("neuralese-recurrence", help="Train/load the recurrent-reasoning organism and run stepwise logit lens + patching against its hidden 'thoughts.'")
    p11.add_argument("--checkpoint", default="checkpoints/recurrent_reasoner.npz")
    p11.add_argument("--steps", type=int, default=5000)
    p11.add_argument("--seed", type=int, default=0)
    p11.add_argument("--retrain", action="store_true")
    p11.set_defaults(func=cmd_neuralese_recurrence)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
