"""Read-only sync_lab RGB-D input and independent RGB rectification.

Depth files are retained as source metadata; the RGB-only pipeline never reads,
resizes or fits them. Rectification math matches sync_lab/hocap_sequence.py.
"""

import hashlib
import json
import os
from pathlib import Path
import tempfile
import cv2
import numpy as np

CONVENTION = "p_master = T_master_from_camera @ p_camera (homogeneous column vectors)"


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def atomic_json(path, value):
    atomic_write(
        path,
        lambda f: f.write(
            (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()
        ),
    )


def atomic_npz(path, **arrays):
    atomic_write(path, lambda f: np.savez_compressed(f, **arrays))


def atomic_write(path, writer):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    name = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as f:
            name = f.name
            writer(f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(name, path)
    finally:
        if name and os.path.exists(name):
            os.unlink(name)


def rigid(value):
    t = np.asarray(value, dtype=float)
    if (
        t.shape != (4, 4)
        or not np.isfinite(t).all()
        or not np.allclose(t[3], [0, 0, 0, 1])
        or not np.allclose(t[:3, :3].T @ t[:3, :3], np.eye(3), atol=1e-5)
        or not np.isclose(np.linalg.det(t[:3, :3]), 1, atol=1e-5)
    ):
        raise ValueError("Invalid camera rigid transform")
    return t


def rectification_maps(intr):
    w, h = intr["width"], intr["height"]
    if any(type(n) != int or n <= 0 for n in (w, h)):
        raise ValueError("Invalid image size")
    focal = np.array([intr["fx"], intr["fy"]])
    center = np.array([intr["ppx"], intr["ppy"]])
    d = np.asarray(intr["coeffs"])
    model = intr["model"].removeprefix("distortion.")
    if (
        d.shape != (5,)
        or not np.isfinite([*focal, *center, *d]).all()
        or min(focal) <= 0
    ):
        raise ValueError("Invalid RGB intrinsics")
    if model not in ("none", "brown_conrady", "inverse_brown_conrady"):
        raise ValueError("Unknown distortion model")
    y, x = np.mgrid[:h, :w]
    xy = (np.stack((x, y), axis=-1) - center) / focal
    if model != "none":
        x, y = xy[..., 0], xy[..., 1]
        k1, k2, p1, p2, k3 = d
        r2 = x * x + y * y
        radial = 1 + r2 * (k1 + r2 * (k2 + r2 * k3))
        xq, yq = x * radial, y * radial
        tx, ty = (xq, yq) if model == "inverse_brown_conrady" else (x, y)
        xy = np.stack(
            (
                xq + 2 * p1 * tx * ty + p2 * (r2 + 2 * tx * tx),
                yq + 2 * p2 * tx * ty + p1 * (r2 + 2 * ty * ty),
            ),
            axis=-1,
        )
    source = xy * focal + center
    if not np.isfinite(source).all():
        raise ValueError("Invalid rectification map")
    return source[..., 0].astype(np.float32), source[..., 1].astype(np.float32)


def source_path(root, relative):
    p = (root / relative).resolve()
    if not p.is_relative_to(root) or not p.is_file():
        raise ValueError(f"Missing/unsafe input path: {relative}")
    return p


def prepare(data_root, workdir, limit=None, replace=False):
    data_root = Path(data_root).resolve()
    root = data_root / "rgbd"
    workdir = Path(workdir).resolve()
    if workdir == data_root or workdir.is_relative_to(root):
        raise ValueError("Output must not modify the original RGB-D directory")
    target = workdir / "inputs"
    if target.exists() and (not replace or (workdir / "final").exists()):
        raise ValueError(
            "inputs already exists; use a new workdir, or --replace only when no final exists"
        )
    extr = read_json(root / "calibration/extrinsics.json")
    if extr.get("convention") != CONVENTION or extr.get("translation_unit") != "meter":
        raise ValueError("Unknown external calibration convention or units")
    rows = [
        json.loads(x)
        for x in (root / "frames.jsonl").read_text().splitlines()
        if x.strip()
    ]
    ids = [str(r["frame_id"]) for r in rows]
    if not rows or len(set(ids)) != len(ids):
        raise ValueError("Empty/duplicate frames")
    if limit is not None:
        if limit <= 0:
            raise ValueError("limit must be positive")
        rows = rows[:limit]
    cameras = {}
    maps = {}
    for p in sorted(root.glob("cam*/calibration.json")):
        meta = read_json(p)
        cid = p.parent.name
        serial = meta["serial"]
        if meta["id"] != cid:
            raise ValueError("Camera ID mismatch")
        intr = meta["original_calibration"]["intrinsics"]["color"]
        maps[cid] = rectification_maps(intr)
        t = rigid(extr["T_master_from_camera"][serial])
        cameras[cid] = dict(
            serial=serial,
            T_master_from_camera=t.tolist(),
            intrinsics=dict(intr, model="none", coeffs=[0.0] * 5),
        )
    if len(cameras) != 6 or len({c["serial"] for c in cameras.values()}) != 6:
        raise ValueError("Expected six distinct cameras")
    master = [c for c in cameras.values() if c["serial"] == extr["master_serial"]]
    if len(master) != 1 or not np.allclose(
        master[0]["T_master_from_camera"], np.eye(4), atol=1e-5
    ):
        raise ValueError("Master camera transform is not identity")
    # Validate all source paths before publishing any input manifest.
    for row in rows:
        if set(row["cameras"]) != set(cameras):
            raise ValueError("Incomplete synchronized camera group")
        for cid, camera in cameras.items():
            item = row["cameras"][cid]
            if item["serial"] != camera["serial"]:
                raise ValueError("Frame camera serial mismatch")
            source_path(root, item["color_path"])
    workdir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".prepare-", dir=workdir) as temp:
        staging = Path(temp) / "inputs"
        staging.mkdir()
        manifest = dict(
            schema="multiview_images_v1",
            units="meter",
            extrinsic_convention=CONVENTION,
            master_serial=extr["master_serial"],
            cameras=cameras,
            frames=[],
        )
        for index, row in enumerate(rows):
            entry = dict(
                frame_id=str(row["frame_id"]),
                images={},
                timestamps_ms={},
                image_sha256={},
            )
            for cid, camera in cameras.items():
                item = row["cameras"][cid]
                image = cv2.imread(str(source_path(root, item["color_path"])))
                intr = camera["intrinsics"]
                if image is None or image.shape != (intr["height"], intr["width"], 3):
                    raise ValueError("RGB size differs from calibration")
                image = cv2.remap(
                    image, *maps[cid], cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT
                )
                relative = f"rgb/{cid}/color_{index:06d}.jpg"
                out = staging / relative
                out.parent.mkdir(parents=True, exist_ok=True)
                if not cv2.imwrite(str(out), image, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                    raise OSError("Failed writing rectified RGB")
                entry["images"][cid] = relative
                entry["image_sha256"][cid] = digest(out)
                entry["timestamps_ms"][cid] = item.get("color_source", {}).get(
                    "timestamp_ms"
                )
            manifest["frames"].append(entry)
        atomic_json(staging / "manifest.json", manifest)
        # Preserve previous inputs until a complete replacement is ready.
        backup = workdir / ".previous_inputs"
        if backup.exists():
            raise ValueError(
                "Previous prepare recovery directory exists; inspect before retry"
            )
        if target.exists():
            target.rename(backup)
        try:
            staging.rename(target)
        except BaseException:
            if backup.exists():
                backup.rename(target)
            raise
        if backup.exists():
            import shutil

            shutil.rmtree(backup)
    return target / "manifest.json"


class PreparedSequence:
    def __init__(self, workdir):
        self.workdir = Path(workdir)
        self.path = self.workdir / "inputs/manifest.json"
        self.root = self.path.parent
        self.manifest = read_json(self.path)
        m = self.manifest
        if (
            m["schema"] != "multiview_images_v1"
            or m["units"] != "meter"
            or m["extrinsic_convention"] != CONVENTION
        ):
            raise ValueError("Unsupported prepared image manifest")
        self.frame_ids = [str(f["frame_id"]) for f in m["frames"]]
        self.camera_ids = list(m["cameras"])
        if len(set(self.frame_ids)) != len(self.frame_ids) or not self.frame_ids:
            raise ValueError("Invalid frame index")
        self.fingerprint = digest(self.path)
        self.matrices = []
        self.sizes = []
        for cid in self.camera_ids:
            c = m["cameras"][cid]
            intr = c["intrinsics"]
            if intr["model"] != "none" or np.any(intr["coeffs"]):
                raise ValueError("RGB pipeline requires prepared rectified images")
            k = np.array(
                [
                    [intr["fx"], 0, intr["ppx"]],
                    [0, intr["fy"], intr["ppy"]],
                    [0, 0, 1.0],
                ]
            )
            self.matrices.append(
                k @ np.linalg.inv(rigid(c["T_master_from_camera"]))[:3]
            )
            self.sizes.append([intr["width"], intr["height"]])
        self.matrices = np.array(self.matrices)
        self.sizes = np.array(self.sizes)

    def image(self, index, cid):
        frame = self.manifest["frames"][index]
        p = source_path(self.root, frame["images"][cid])
        if digest(p) != frame["image_sha256"][cid]:
            raise ValueError("Prepared RGB changed since manifest creation")
        bgr = cv2.imread(str(p))
        if bgr is None or bgr.shape[1::-1] != tuple(
            self.sizes[self.camera_ids.index(cid)]
        ):
            raise ValueError("Invalid prepared RGB")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def load(self, name):
        p = self.workdir / "predictions" / name
        with np.load(p, allow_pickle=False) as z:
            data = {k: z[k] for k in z.files}
        if (
            str(data["manifest_sha256"]) != self.fingerprint
            or data["frame_ids"].tolist() != self.frame_ids
        ):
            raise ValueError(f"{name} is stale or belongs to other frames")
        return data

    def common(self):
        return dict(
            frame_ids=np.array(self.frame_ids),
            camera_ids=np.array(self.camera_ids),
            manifest_sha256=np.array(self.fingerprint),
        )
