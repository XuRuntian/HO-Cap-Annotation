"""Import EasyMocap trial_shape.npz as a reusable, verified fixed MANO shape."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

import numpy as np


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_shape(path):
    with np.load(path, allow_pickle=False) as data:
        betas = np.asarray(data['betas'], dtype=np.float64)
        order = data['hand_order'].tolist()
    if betas.shape != (2, 10) or not np.isfinite(betas).all():
        raise ValueError('betas must be finite [2,10]')
    if not isinstance(order, list) or sorted(order) != ['left', 'right']:
        raise ValueError('hand_order must contain right and left exactly once')
    return betas[[order.index('right'), order.index('left')]]


def verify_models(easymocap_root, easymocap_python, mano_root):
    # Inspect effective model buffers, not just filenames: loaders may alter bases.
    script = '''
import sys
import numpy as np
from easymocap.smplmodel import load_model
out = {}
for side, kind in [('right', 'manor'), ('left', 'manol')]:
    model = load_model(model_type=kind, use_cuda=False)
    for key in ['v_template', 'shapedirs']:
        out[side + '_' + key] = getattr(model, key).detach().cpu().numpy()
np.savez(sys.argv[1], **out)
'''
    import os
    from manopth.manolayer import ManoLayer

    env = dict(os.environ)
    env.pop('LD_LIBRARY_PATH', None)
    env.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    errors = {}
    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp) / 'buffers.npz'
        subprocess.run([str(easymocap_python), '-c', script, str(target)],
                       cwd=easymocap_root, env=env, check=True)
        with np.load(target, allow_pickle=False) as source:
            for side in ('right', 'left'):
                model = ManoLayer(mano_root=str(mano_root), side=side,
                                  use_pca=True, ncomps=45, flat_hand_mean=False)
                for name in ('v_template', 'shapedirs'):
                    actual = getattr(model, 'th_' + name).detach().cpu().numpy()
                    if name == 'v_template':
                        actual = actual[0]
                    expected = source[side + '_' + name]
                    if actual.shape != expected.shape:
                        raise ValueError(f'{side} {name}: incompatible shape basis')
                    delta = float(np.max(np.abs(actual - expected)))
                    if not np.isfinite(delta) or delta > 1e-7:
                        raise ValueError(f'{side} {name}: incompatible basis ({delta})')
                    errors[side + '_' + name] = delta
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('--output', type=Path, required=True,
                        help='New directory containing betas.json and metadata.json')
    parser.add_argument('--easymocap-root', type=Path, required=True)
    parser.add_argument('--easymocap-python', type=Path, required=True)
    parser.add_argument('--mano-root', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output already exists; choose a new directory')
    betas = read_shape(args.source)
    errors = verify_models(args.easymocap_root.resolve(),
                           args.easymocap_python.resolve(), args.mano_root.resolve())
    metadata = {
        'schema': 'hocap.fixed_hand_shape.v1',
        'hand_order': ['right', 'left'],
        'source': str(args.source.resolve()),
        'source_sha256': sha256(args.source),
        'loader': 'easymocap.smplmodel.load_model',
        'basis_max_abs_error_m': errors,
        'model_sha256': {side: sha256(args.mano_root / f'MANO_{side.upper()}.pkl')
                         for side in ('right', 'left')},
        'max_abs_beta': np.abs(betas).max(axis=1).tolist(),
        'quality_status': 'trial_not_manually_accepted',
        'note': 'Compatibility is not a quality guarantee. Reuse only for the same person.',
    }
    report = args.source.with_name('report.json')
    if report.exists():
        metadata['source_report_sha256'] = sha256(report)
    args.output.mkdir(parents=True, exist_ok=False)
    target = args.output / 'betas.json'
    target.write_text(json.dumps(betas.tolist(), indent=2) + '\n')
    metadata['betas_sha256'] = sha256(target)
    (args.output / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(target.resolve())


if __name__ == '__main__':
    main()
