"""Owned classification links and a compact, atomic CSV index."""

import csv
import fcntl
import io
import os
import re
import uuid
from pathlib import Path

from .batch_inputs import fingerprint
from .rgbd_data import atomic_json, atomic_write, read_json

CATEGORIES = ("01_normal", "02_degraded", "03_needs_review", "04_failed")
COLUMNS = (
    "sequence",
    "status",
    "excluded_cameras",
    "total_frames",
    "right_valid_frames",
    "left_valid_frames",
    "reason",
)


def sequence_id(path):
    path = Path(path).resolve()
    label = re.sub(r"[^a-zA-Z0-9_-]", "_", path.name)[:48] or "sequence"
    return label + "-" + fingerprint(str(path))[:24]


def real_directory(path):
    path = Path(path)
    for part in (path, *path.parents):
        if part.is_symlink():
            raise ValueError(f"Symlink output directory refused: {part}")
    path.mkdir(parents=True, exist_ok=True)
    if not path.is_dir():
        raise ValueError(f"Not a directory: {path}")


class BatchResults:
    def __init__(self, root):
        self.root = Path(root).absolute()
        real_directory(self.root)
        self.fd = os.open(self.root, os.O_RDONLY)
        try:
            fcntl.flock(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.path = self.root / "batch_index.json"
            if self.path.is_symlink():
                raise ValueError("Unsafe batch index")
            if self.path.exists():
                self.index = read_json(self.path)
                if self.index.get("schema") != "hocap.batch_results.v1":
                    raise ValueError("Not an owned batch index")
            else:
                if any(
                    (self.root / n).exists() or (self.root / n).is_symlink()
                    for n in ("summary.csv", "details", "work")
                ):
                    raise ValueError(
                        "Unowned batch artifacts already exist; choose a new batch_results"
                    )
                self.index = dict(schema="hocap.batch_results.v1", entries={})
                atomic_json(self.path, self.index)
            for name in (*CATEGORIES, "details"):
                real_directory(self.root / name)
            if (self.root / "summary.csv").is_symlink():
                raise ValueError("Unsafe summary path")
            for entry in self.index["entries"].values():
                if "transition" in entry:
                    self._finish_transition(entry)
                for pending in entry.pop("pending_cleanup", []):
                    old = self._owned_link(pending)
                    if old is not None:
                        old.unlink()
            atomic_json(self.path, self.index)
            self.write_summary()
        except BaseException:
            os.close(self.fd)
            raise

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        os.close(self.fd)

    def details(self, source):
        self.register(source)
        sid = sequence_id(source)
        path = self.root / "details" / sid
        entry = self.index["entries"][sid]
        if not entry.get("details_owned"):
            if path.exists() or path.is_symlink():
                raise ValueError("Unowned sequence details directory exists")
            real_directory(path)
            entry["details_owned"] = True
            atomic_json(self.path, self.index)
        else:
            real_directory(path)
        return path

    def register(self, source):
        sid = sequence_id(source)
        if sid not in self.index["entries"]:
            self.index["entries"][sid] = dict(
                source=str(Path(source).resolve()), details_owned=False
            )
            atomic_json(self.path, self.index)

    def _owned_link(self, entry, *, staging=False):
        relative = entry.get("link")
        if not relative:
            return None
        path = self.root / relative
        allowed = (
            (
                path.parent == self.root
                and re.fullmatch(r"\.link-[0-9a-f]{32}", path.name)
            )
            if staging
            else (path.parent.name in CATEGORIES and path.parent.parent == self.root)
        )
        if not allowed:
            raise ValueError("Invalid owned link record")
        if not path.is_symlink():
            return None
        st = path.lstat()
        if [st.st_dev, st.st_ino] != entry.get("link_identity"):
            return None
        if os.readlink(path) != entry["source"]:
            return None
        return path

    def _finish_transition(self, entry):
        transition = entry["transition"]
        new = transition["new"]
        target = self.root / new["link"]
        owned = self._owned_link(new)
        staging = self._owned_link(transition["staging"], staging=True)
        if owned is None:
            if target.exists() or target.is_symlink() or staging is None:
                if staging is not None:
                    staging.unlink()
                entry.pop("transition")
                atomic_json(self.path, self.index)
                return False
            # Hard-link the symlink inode: ownership is recorded before publication.
            os.link(staging, target, follow_symlinks=False)
        entry.update(new, row=transition["row"])
        atomic_json(self.path, self.index)
        previous = self._owned_link(transition["previous"])
        if previous is not None and previous != target:
            previous.unlink()
        if staging is not None:
            staging.unlink()
        entry.pop("transition")
        atomic_json(self.path, self.index)
        return True

    def update(self, source, row):
        if row["status"] not in CATEGORIES:
            raise ValueError("Unknown final category")
        self.register(source)
        sid = sequence_id(source)
        entry = self.index["entries"][sid]
        if "transition" in entry:
            self._finish_transition(entry)
        old = self._owned_link(entry)
        target = self.root / row["status"] / sid
        if target != old and (target.exists() or target.is_symlink()):
            raise ValueError(f"Classification path occupied by unowned file: {target}")
        if target == old:
            entry["row"] = {key: row.get(key, "") for key in COLUMNS}
            atomic_json(self.path, self.index)
        else:
            staging = self.root / (".link-" + uuid.uuid4().hex)
            staging.symlink_to(entry["source"], target_is_directory=True)
            st = staging.lstat()
            identity = [st.st_dev, st.st_ino]
            entry["transition"] = dict(
                new=dict(
                    source=entry["source"],
                    link=str(target.relative_to(self.root)),
                    link_identity=identity,
                ),
                staging=dict(
                    source=entry["source"], link=staging.name, link_identity=identity
                ),
                previous={
                    k: entry[k]
                    for k in ("source", "link", "link_identity")
                    if k in entry
                },
                row={key: row.get(key, "") for key in COLUMNS},
            )
            atomic_json(self.path, self.index)
            if not self._finish_transition(entry):
                raise ValueError("Classification target changed during publication")
        self.write_summary()

    def failed_without_link(self, source, row):
        """A user-owned collision cannot be fixed by deleting or replacing it."""
        entry = self.index["entries"][sequence_id(source)]
        if "transition" in entry:
            self._finish_transition(entry)
        old = self._owned_link(entry)
        if old is not None:
            old.unlink()
        entry.pop("link", None)
        entry.pop("link_identity", None)
        entry["row"] = row
        atomic_json(self.path, self.index)
        self.write_summary()

    def write_summary(self):
        stream = io.StringIO(newline="")
        writer = csv.DictWriter(stream, fieldnames=COLUMNS)
        writer.writeheader()
        for key in sorted(self.index["entries"]):
            item = self.index["entries"][key]
            if "row" in item:
                writer.writerow(item["row"])
        atomic_write(
            self.root / "summary.csv",
            lambda f: f.write(stream.getvalue().encode("utf-8")),
        )
