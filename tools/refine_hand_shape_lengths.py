"""Refine a fixed MANO hand shape using metric MANO landmark distances."""

import argparse
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import least_squares

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hocap_annotation.hand_shape import load_hand_shape, model_basis_hashes
from hocap_annotation.hand_surface_measurements import NIOSH_METRICS, measure_surface, neutral_surface_geometry
from hocap_annotation.rgbd_data import atomic_json, digest, read_json

SIDES = ("right", "left")
FINGERS = dict(thumb=1, index=5, middle=9, ring=13, little=17)
SEGMENTS = {
    f"{finger}_{segment}": ((root+i, root+i+1),)
    for finger, root in FINGERS.items()
    for i, segment in enumerate(("proximal", "middle", "distal"))
}
# Thumb landmarks are CMC, MCP, IP, tip rather than three phalanges.
SEGMENTS["thumb_cmc_to_mcp"] = SEGMENTS.pop("thumb_proximal")
SEGMENTS["thumb_mcp_to_ip"] = SEGMENTS.pop("thumb_middle")
SEGMENTS["thumb_ip_to_tip"] = SEGMENTS.pop("thumb_distal")
PALM = {"palm_length": ((0, 9),), "palm_width": ((5, 17),)}
TOTALS = {f"{finger}_total": tuple((r+i, r+i+1) for i in range(3))
          for finger, r in FINGERS.items()}
METRICS = {**PALM, **SEGMENTS, **TOTALS}
BONE_DEFINITION = "mano_landmark_v1"
SURFACE_DEFINITION = "niosh_external_v1"


def parse_targets(raw):
    if (not isinstance(raw, dict) or set(raw) - {"units", "definition", *SIDES}
            or raw.get("units") != "mm"):
        raise ValueError("Expected an object with units='mm' and right/left targets")
    definition = raw.get("definition", BONE_DEFINITION)
    if definition not in (BONE_DEFINITION, SURFACE_DEFINITION):
        raise ValueError(f"Unknown measurement definition: {definition!r}")
    allowed = NIOSH_METRICS if definition == SURFACE_DEFINITION else METRICS
    result = {}
    for side in SIDES:
        entries = raw.get(side, {})
        if not isinstance(entries, dict):
            raise ValueError(f"{side}: expected an object of metric names and lengths")
        filled = {}
        for name, length in entries.items():
            if name not in allowed:
                raise ValueError(f"{side}: unknown metric {name!r}")
            if length is None:
                continue
            if (isinstance(length, bool) or not isinstance(length, (int, float))
                    or not np.isfinite(length) or length <= 0):
                raise ValueError(f"{side}/{name}: expected a positive finite value in mm")
            filled[name] = float(length)
        if filled:
            result[side] = filled
    if not result and definition == BONE_DEFINITION:
        raise ValueError("Provide at least one measurement")
    return definition, result


def read_targets(path):
    if path.suffix.lower() not in (".yaml", ".yml"):
        return parse_targets(read_json(path))
    from ruamel.yaml import YAML, YAMLError

    try:
        raw = YAML(typ="safe").load(path.read_text(encoding="utf-8"))
    except YAMLError as exc:
        raise ValueError(f"Invalid measurement YAML: {exc}") from exc
    return parse_targets(raw)


def neutral_geometry(mano, side):
    import torch
    from manopth.manolayer import ManoLayer

    layer = ManoLayer(mano_root=str(mano), side=side, use_pca=False,
                      flat_hand_mean=True).double()
    with torch.no_grad():
        _, joints = layer(torch.zeros((11, 48), dtype=torch.float64),
                          torch.cat((torch.zeros((1, 10)), torch.eye(10))).double())
    neutral, *samples = joints.numpy()
    basis = np.asarray(samples) - neutral

    def predict(beta):
        return neutral + np.einsum("k,kjc->jc", beta, basis)

    return predict


def refine(initial, predict, targets, max_delta):
    def evaluate(beta):
        points = predict(beta)
        return {name: sum(np.linalg.norm(points[a] - points[b]) for a, b in edges)
                for name, edges in METRICS.items()}

    requested = {k: METRICS[k] for k in targets}
    targeted_edges = {edge for edges in requested.values() for edge in edges}
    protected = {k: v for k, v in {**PALM, **SEGMENTS}.items()
                 if not any(edge in targeted_edges for edge in v)}
    return solve(initial, evaluate, targets, tuple(protected), max_delta)


def solve(initial, evaluate, targets, protected, max_delta, *, report_only=()):
    def select(values, keys):
        return np.array([values[key] for key in keys])

    baseline = evaluate(initial)
    guard = select(baseline, protected)
    active = {name: value for name, value in targets.items() if name not in report_only}
    target = np.array(list(active.values()))

    def residual(beta):
        values = evaluate(beta)
        return np.r_[3.5 * (select(values, active) - target),
                     1.2 * (select(values, protected) - guard), 2 * (beta-initial)]

    opt = least_squares(residual, initial, bounds=(initial-max_delta, initial+max_delta),
                        loss="soft_l1", f_scale=2, max_nfev=160)
    if not opt.success or not np.isfinite(opt.x).all():
        raise ValueError(f"Shape refinement failed: {opt.message}")
    # Report geometry for the actual serialized coefficients, not the float64 optimizer state.
    refined = opt.x.astype(np.float32)
    final = evaluate(refined)
    before, after = select(baseline, targets), select(final, targets)
    drift = float(np.max(np.abs(select(final, protected)-guard), initial=0))
    if drift > 2:
        raise ValueError(f"Refinement moves an unrequested length by {drift:.2f} mm; "
                         "check the measurements or reduce --max-beta-delta")
    return refined, dict(
        target_mm={name: dict(requested=value, before=float(a), after=float(b),
                             error_after=float(b-value), used_in_fit=name not in report_only)
                   for (name, value), a, b in zip(targets.items(), before, after)},
        max_untargeted_length_change_mm=drift,
        max_beta_change=float(np.max(np.abs(refined-initial))),
    )


def refine_surface(initial, mesh, targets, max_delta):
    parts = {"palm_length", "middle_length"}
    report_only = ("hand_length",) if (parts | {"hand_length"}) <= targets.keys() else ()
    protected = set(NIOSH_METRICS) - targets.keys()
    if parts & targets.keys():
        protected.discard("hand_length")
    if "hand_length" in targets:
        protected -= parts
    return solve(initial, lambda b: measure_surface(mesh(b)), targets,
                 tuple(name for name in NIOSH_METRICS if name in protected), max_delta,
                 report_only=report_only)


def run(source, measurements, output, mano, max_delta=1.5):
    if not np.isfinite(max_delta) or not 0 < max_delta <= 3:
        raise ValueError("--max-beta-delta must be in (0, 3]")
    source = source.expanduser().resolve()
    base = source.parent if source.is_file() else source
    beta = np.asarray(load_hand_shape(source, mano), dtype=np.float32)
    definition, targets = read_targets(measurements)
    if not targets:
        print("No measurements supplied; keeping the input hand shape.", file=sys.stderr)
        return base / "betas.json"
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Choose a new output directory: {output}")
    report = {}
    for side, target in targets.items():
        i = SIDES.index(side)
        initial = beta[i].astype(float)
        if definition == SURFACE_DEFINITION:
            mesh = neutral_surface_geometry(mano, side)
            beta[i], report[side] = refine_surface(initial, mesh, target, max_delta)
        else:
            beta[i], report[side] = refine(initial, neutral_geometry(mano, side),
                                          target, max_delta)
    meta = read_json(base / "metadata.json")
    meta.update(schema="hocap.fixed_hand_shape.v2", basis_encoding="float32-le-c-order",
                basis_sha256=model_basis_hashes(mano),
                method="Optional bounded MANO dimension refinement",
                measurement_definition=definition,
                quality_status="trial_not_manually_accepted",
                source_hand_shape=str(base), source_betas_sha256=digest(base / "betas.json"),
                measurements_sha256=digest(measurements), refined_sides=list(report))
    output.mkdir(parents=True, exist_ok=False)
    beta_path = output / "betas.json"
    atomic_json(beta_path, beta.tolist())
    atomic_json(output / "refinement.json", dict(units="mm", definition=definition, sides=report,
                                               max_beta_delta=max_delta))
    meta.update(betas_sha256=digest(beta_path),
                refinement_sha256=digest(output / "refinement.json"))
    atomic_json(output / "metadata.json", meta)
    return beta_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("hand_shape", type=Path, help="Input bundle directory or betas.json")
    parser.add_argument("measurements", type=Path, help="YAML or JSON file with target distances in mm")
    parser.add_argument("output", type=Path, help="New output bundle directory")
    parser.add_argument("--mano-root", type=Path, default=Path.home() / "models/mano")
    parser.add_argument("--max-beta-delta", type=float, default=1.5)
    args = parser.parse_args()
    try:
        print(run(args.hand_shape, args.measurements, args.output,
                  args.mano_root, args.max_beta_delta))
    except (ValueError, OSError, KeyError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
