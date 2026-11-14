"""Content-checked stage receipts, never 'file exists' resume decisions."""

import fcntl
import os
from pathlib import Path

from .batch_inputs import fingerprint
from .batch_results import real_directory
from .rgbd_data import atomic_json, digest, read_json


class StaleStage(ValueError):
    pass


class StageCache:
    def __init__(self, workdir, source, *, readonly=False):
        self.workdir = Path(workdir).absolute()
        self.path = self.workdir / "batch_state.json"
        self.readonly = readonly
        self.fd = None
        if readonly:
            self.fd = os.open(self.workdir, os.O_RDONLY)
            try:
                fcntl.flock(self.fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
            except BaseException:
                os.close(self.fd)
                self.fd = None
                raise
        if not readonly:
            real_directory(self.workdir)
            if (self.workdir / "final").exists() or (
                self.workdir / "final"
            ).is_symlink():
                raise ValueError("Workdir has final/; choose an independent work root")
            for name in ("inputs", "predictions", "batch_state.json"):
                if (self.workdir / name).is_symlink():
                    raise ValueError(f"Symlink work artifact refused: {name}")
            for parent in ("inputs", "predictions"):
                directory = self.workdir / parent
                if directory.exists() and any(
                    p.is_symlink() for p in directory.rglob("*")
                ):
                    raise ValueError(f"Symlink inside work artifacts refused: {parent}")
            self.fd = os.open(self.workdir, os.O_RDONLY)
            try:
                fcntl.flock(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BaseException:
                os.close(self.fd)
                raise
        self.state = dict(
            schema="hocap.batch_stages.v1",
            source=str(Path(source).resolve()),
            stages={},
        )
        try:
            if self.path.is_symlink():
                raise ValueError("Unsafe stage receipt")
            if self.path.exists():
                state = read_json(self.path)
                if (
                    state.get("schema") != self.state["schema"]
                    or state.get("source") != self.state["source"]
                ):
                    raise ValueError("Workdir belongs to another sequence or tool")
                self.state = state
        except BaseException:
            self.close()
            raise

    def close(self):
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def _hashes(self, outputs):
        result = {}
        for relative in outputs:
            path = self.workdir / relative
            if not path.is_relative_to(self.workdir) or ".." in Path(relative).parts:
                raise ValueError("Unsafe stage output")
            for p in (path, *path.parents):
                if p == self.workdir:
                    break
                if p.is_symlink():
                    raise ValueError(f"Symlink stage output refused: {p}")
            result[relative] = digest(path)
        return result

    def valid(self, stage, signature, outputs):
        receipt = self.state["stages"].get(stage, {})
        try:
            return receipt.get("signature") == fingerprint(signature) and receipt.get(
                "outputs"
            ) == self._hashes(outputs)
        except (OSError, ValueError):
            return False

    def record(self, stage, signature, outputs):
        if self.readonly:
            raise ValueError("Read-only stage cache")
        self.state["stages"][stage] = dict(
            signature=fingerprint(signature),
            inputs=signature,
            outputs=self._hashes(outputs),
        )
        atomic_json(self.path, self.state)

    def run(self, stage, signature, outputs, action, *, resume=False, replace=False):
        if self.valid(stage, signature, outputs) and resume:
            return "reused"
        existing = any(
            (self.workdir / p).exists() or (self.workdir / p).is_symlink()
            for p in outputs
        )
        if existing and not replace:
            raise StaleStage(
                f"{stage}: unverified/changed result or resume not requested; use new work root or --replace"
            )
        # Check output boundaries even when the files do not exist yet.
        for p in outputs:
            path = self.workdir / p
            for parent in (path, *path.parents):
                if parent == self.workdir:
                    break
                if parent.is_symlink():
                    raise ValueError("Symlink stage output refused")
        if self.readonly:
            raise StaleStage(f"{stage}: no verified cached result")
        action()
        self.record(stage, signature, outputs)
        return "ran"
