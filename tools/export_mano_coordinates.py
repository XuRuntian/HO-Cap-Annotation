"""Export fitted MANO joints/vertices in master and table coordinates."""

import os
os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from hocap_annotation.loaders import SequenceLoader


def rigid(value):
    matrix = np.asarray(value, dtype=float)
    if (matrix.shape != (4, 4) or not np.isfinite(matrix).all()
            or not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-6)
            or not np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-5)
            or not np.isclose(np.linalg.det(matrix[:3, :3]), 1, atol=1e-5)):
        raise ValueError('Expected a finite rigid transform')
    return matrix


def transform_points(points, matrix):
    return (points @ matrix[:3, :3].T + matrix[:3, 3]).astype(np.float32)


def export_coordinates(sequence, poses_file, table_file, output, confirm_axes=False,
                       allow_rig_warning=False, device='cuda', batch_size=128,
                       rig_report_file=None, require_rig_pass=False):
    output = Path(output)
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite {output}')
    if not confirm_axes:
        raise ValueError('Inspect table-axis previews, then explicitly pass --confirm-table-axes')
    table = json.loads(Path(table_file).read_text())
    if table.get('translation_unit') != 'meter':
        raise ValueError('Expected table calibration in meters')
    # Coordinate conversion and rig quality are independent operations. Legacy
    # table-file warnings remain in provenance, but are not a conversion gate.
    if allow_rig_warning:
        print('NOTE: --allow-rig-warning is deprecated; use a separate --rig-report and optional --require-rig-pass.')
    rig_report = json.loads(Path(rig_report_file).read_text()) if rig_report_file else None
    if rig_report is not None:
        if rig_report.get('schema') != 'sync_lab.rig_validation.v1':
            raise ValueError('Expected an independent rig-validation report')
        if rig_report.get('master_serial') != table['master_serial']:
            raise ValueError('Rig report and table use different master cameras')
    if require_rig_pass and (rig_report is None or rig_report.get('status') != 'consistent'):
        raise ValueError('--require-rig-pass needs a matching rig report with status consistent')
    table_from_master = rigid(table['T_table_from_master'])
    if not np.allclose(table_from_master @ rigid(table['T_master_from_table']), np.eye(4), atol=1e-6):
        raise ValueError('Table transform directions are inconsistent')
    if batch_size < 1:
        raise ValueError('batch_size must be positive')
    loader = SequenceLoader(sequence, load_mano=True, device=device)
    if table['master_serial'] != loader.rs_master:
        raise ValueError('Master camera does not match table calibration')
    reference_rig = table.get('rig_T_master_from_camera', {})
    rig_matches_table_snapshot = True
    for serial, pose in zip(loader.rs_serials, loader.extrinsics2master.cpu().numpy()):
        rig_matches_table_snapshot &= serial in reference_rig and np.allclose(
            pose, rigid(reference_rig[serial]), atol=1e-5)
        if rig_report is not None:
            checked_rig = rig_report['rig_T_master_from_camera']
            if serial not in checked_rig or not np.allclose(pose, rigid(checked_rig[serial]), atol=1e-5):
                raise ValueError(f'Rig quality report does not match sequence extrinsics: {serial}')
    if not rig_matches_table_snapshot:
        print('NOTE: Rig extrinsics differ from the table snapshot. Table transform assumes the master did not move relative to the table.')
    poses_file = Path(poses_file)
    poses = np.load(poses_file, allow_pickle=False)
    if poses.shape != (2, loader.num_frames, 51):
        raise ValueError('Expected canonical right/left poses[2, num_frames, 51]')
    valid = np.isfinite(poses).all(axis=-1) & ~np.all(poses == -1, axis=-1)
    mask_file = poses_file.with_name(poses_file.stem+'_valid.npy')
    if mask_file.exists():
        mask = np.load(mask_file, allow_pickle=False)
        if mask.shape != valid.shape or mask.dtype != bool:
            raise ValueError('Pose validity sidecar has wrong shape/type')
        valid &= mask
    ids = [0 if side == 'right' else 1 for side in loader.mano_sides]
    for hand in set(range(2))-set(ids):
        valid[hand] = False
    safe = np.where(valid[..., None], poses, 0).astype(np.float32)
    joints = np.full((2, loader.num_frames, 21, 3), np.nan, np.float32)
    vertices = np.full((2, loader.num_frames, 778, 3), np.nan, np.float32)
    master_from_world = rigid(loader.tag_1.cpu().numpy())
    with torch.no_grad():
        for start in range(0, loader.num_frames, batch_size):
            stop = min(start+batch_size, loader.num_frames)
            parameters = np.concatenate([safe[i, start:stop] for i in ids], axis=1)
            v, j = loader.mano_group_layer(torch.from_numpy(parameters).to(device))
            v = transform_points(v.cpu().numpy(), master_from_world)
            j = transform_points(j.cpu().numpy(), master_from_world)
            for local, canonical in enumerate(ids):
                joints[canonical, start:stop] = j[:, local*21:(local+1)*21]
                vertices[canonical, start:stop] = v[:, local*778:(local+1)*778]
    joints[~valid] = np.nan
    vertices[~valid] = np.nan
    output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation avoids accidentally replacing another export.
    with output.open('xb') as stream:
        np.savez_compressed(stream, poses_world=poses, valid=valid,
            joints_master=joints, vertices_master=vertices,
            joints_table=transform_points(joints, table_from_master),
            vertices_table=transform_points(vertices, table_from_master),
            betas=loader.mano_beta.cpu().numpy(), hands=np.array(['right', 'left']),
            coordinate_unit=np.asarray('meter'), master_serial=np.asarray(loader.rs_master),
            T_master_from_world=master_from_world, T_table_from_master=table_from_master,
            table_axes_user_confirmed=np.asarray(confirm_axes),
            rig_validation_status=np.asarray(rig_report['status'] if rig_report else 'not_provided'),
            rig_validation_required=np.asarray(require_rig_pass),
            legacy_table_rig_warning=np.asarray(bool(table.get('rig_validation_warning'))),
            rig_matches_table_snapshot=np.asarray(rig_matches_table_snapshot),
            source_rig_report=np.asarray(str(Path(rig_report_file).resolve()) if rig_report_file else ''),
            source_poses=np.asarray(str(poses_file.resolve())), source_table=np.asarray(str(Path(table_file).resolve())))
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sequence_folder', required=True)
    parser.add_argument('--poses', required=True, help='Final poses_m.npy, not 3D keypoints')
    parser.add_argument('--table-frame', required=True, help='table_frame.json with explicit transforms')
    parser.add_argument('--output', required=True, help='New .npz file')
    parser.add_argument('--confirm-table-axes', action='store_true')
    parser.add_argument('--rig-report', help='Optional independent report from check_rig_calibration.py')
    parser.add_argument('--require-rig-pass', action='store_true', help='Quality gate: require a matching report with status consistent')
    parser.add_argument('--allow-rig-warning', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--device', default='cuda', choices=['cpu', 'cuda'])
    parser.add_argument('--batch-size', type=int, default=128)
    args = parser.parse_args()
    output = export_coordinates(args.sequence_folder, args.poses, args.table_frame, args.output,
                                args.confirm_table_axes, args.allow_rig_warning, args.device, args.batch_size,
                                args.rig_report, args.require_rig_pass)
    print(output.resolve())


if __name__ == '__main__':
    main()
