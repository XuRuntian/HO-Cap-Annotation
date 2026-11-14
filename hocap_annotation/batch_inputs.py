"""Read-only RGB input validation and content fingerprints for resumable batches."""

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from .rgbd_data import (
    CONVENTION,
    digest,
    read_json,
    rectification_maps,
    rigid,
    source_path,
)


def fingerprint(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, allow_nan=False, separators=(",", ":")
        ).encode()
    ).hexdigest()


class InputError(ValueError):
    pass


def inspect_inputs(sequence, policy):
    """Check every RGB image, not just the geometric sample. Never change sources."""
    root = Path(sequence).resolve() / "rgbd"
    try:
        extr_path = root / "calibration/extrinsics.json"
        frames_path = root / "frames.jsonl"
        extr = read_json(extr_path)
        if not isinstance(extr, dict):
            raise InputError("extrinsics must be a JSON object")
        if (
            extr.get("convention") != CONVENTION
            or extr.get("translation_unit") != "meter"
        ):
            raise InputError("unknown extrinsic convention or units")
        rows = [
            json.loads(line)
            for line in frames_path.read_text().splitlines()
            if line.strip()
        ]
        ids = [str(r["frame_id"]) for r in rows]
        if not rows or len(set(ids)) != len(ids):
            raise InputError("empty or duplicate frame IDs")
        cameras, maps = {}, {}
        files = {str(p.relative_to(root)): digest(p) for p in (extr_path, frames_path)}
        for p in sorted(root.glob("cam*/calibration.json")):
            meta = read_json(p)
            cid, serial = p.parent.name, meta["serial"]
            if meta["id"] != cid:
                raise InputError("calibration camera ID mismatch")
            intr = meta["original_calibration"]["intrinsics"]["color"]
            if not isinstance(intr, dict) or not isinstance(intr.get("model"), str):
                raise InputError("invalid RGB intrinsics object/model")
            maps[cid] = rectification_maps(intr)
            cameras[cid] = dict(
                serial=serial,
                T_master_from_camera=rigid(
                    extr["T_master_from_camera"][serial]
                ).tolist(),
                intrinsics=dict(intr, model="none", coeffs=[0.0] * 5),
            )
            files[str(p.relative_to(root))] = digest(p)
        if len(cameras) != 6 or len({c["serial"] for c in cameras.values()}) != 6:
            raise InputError("expected six distinct calibrated cameras")
        master = [c for c in cameras.values() if c["serial"] == extr["master_serial"]]
        if len(master) != 1 or not np.allclose(
            master[0]["T_master_from_camera"], np.eye(4), atol=1e-5
        ):
            raise InputError("master transform must be identity")
        manifest = dict(
            schema="multiview_images_v1",
            units="meter",
            extrinsic_convention=CONVENTION,
            master_serial=extr["master_serial"],
            cameras=cameras,
            frames=[],
        )
        previous, used, spans = {}, set(), []
        for index, row in enumerate(rows):
            if set(row["cameras"]) != set(cameras):
                raise InputError(
                    f"frame {ids[index]}: incomplete camera correspondence"
                )
            frame = dict(
                frame_id=ids[index], images={}, timestamps_ms={}, image_sha256={}
            )
            for cid, camera in cameras.items():
                item = row["cameras"][cid]
                if item["serial"] != camera["serial"]:
                    raise InputError(f"{cid}: frame serial mismatch")
                color_source = item.get("color_source")
                if not isinstance(color_source, dict):
                    raise InputError("color_source must be a JSON object")
                stamp = color_source.get("timestamp_ms")
                if (
                    isinstance(stamp, bool)
                    or not isinstance(stamp, (int, float))
                    or not np.isfinite(stamp)
                    or stamp <= previous.get(cid, -np.inf)
                ):
                    raise InputError(
                        f"{cid} frame {ids[index]}: missing/non-increasing RGB timestamp"
                    )
                if (
                    cid in previous
                    and round(
                        stamp - rows[0]["cameras"][cid]["color_source"]["timestamp_ms"]
                    )
                    <= previous[(cid, "elapsed")]
                ):
                    raise InputError(
                        f"{cid}: timestamps collide at millisecond precision"
                    )
                previous[cid] = stamp
                previous[(cid, "elapsed")] = round(
                    stamp - rows[0]["cameras"][cid]["color_source"]["timestamp_ms"]
                )
                path = source_path(root, item["color_path"])
                if path in used or path.stat().st_size == 0:
                    raise InputError(
                        f"duplicate or empty RGB file: {item['color_path']}"
                    )
                used.add(path)
                image = cv2.imread(str(path))
                intr = camera["intrinsics"]
                if (
                    image is None
                    or image.shape != (intr["height"], intr["width"], 3)
                    or not image.any()
                ):
                    raise InputError(
                        f"empty/unreadable/wrong-size RGB: {item['color_path']}"
                    )
                files[str(path.relative_to(root))] = digest(path)
                corrected = cv2.remap(
                    image, *maps[cid], cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT
                )
                ok, encoded = cv2.imencode(
                    ".jpg", corrected, [cv2.IMWRITE_JPEG_QUALITY, 95]
                )
                if not ok:
                    raise InputError("RGB encoding failed")
                frame["images"][cid] = f"rgb/{cid}/color_{index:06d}.jpg"
                frame["timestamps_ms"][cid] = stamp
                frame["image_sha256"][cid] = hashlib.sha256(encoded).hexdigest()
            span = max(frame["timestamps_ms"].values()) - min(
                frame["timestamps_ms"].values()
            )
            if span > policy.max_timestamp_span_ms:
                raise InputError(
                    f"frame {ids[index]}: RGB timestamp span {span:.3f} ms exceeds limit"
                )
            spans.append(span)
            manifest["frames"].append(frame)
        return dict(
            source_sha256=fingerprint(files),
            expected_manifest=manifest,
            total_frames=len(rows),
            rgb_files=len(used),
            timestamp_max_span_ms=max(spans),
            warnings=[],
        )
    except (OSError, KeyError, TypeError, ValueError) as exc:
        if isinstance(exc, InputError):
            raise
        raise InputError(str(exc)) from exc


def verify_prepared(seq, inspection):
    if seq.manifest != inspection["expected_manifest"]:
        raise InputError(
            "prepared manifest does not match current raw images/calibration/timestamps"
        )
    for frame in seq.manifest["frames"]:
        for cid, relative in frame["images"].items():
            path = source_path(seq.root, relative)
            if digest(path) != frame["image_sha256"][cid]:
                raise InputError("prepared image content changed")
