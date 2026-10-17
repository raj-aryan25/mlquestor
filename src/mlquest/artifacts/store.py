"""Content-addressed, immutable artifact storage on the local filesystem."""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Callable, Sequence, cast

from mlquest.domain import (
    ArtifactKind,
    ArtifactLineage,
    ArtifactManifest,
    DatasetRole,
    DerivedArtifact,
    SourceArtifact,
    StableId,
)
from mlquest.domain.common import validate_identifier

_CHUNK_SIZE = 1024 * 1024
_SHA256_PATTERN = re.compile(r"^[0-9a-fA-F]{64}$")


class ArtifactStoreError(Exception):
    """Base class for artifact store failures."""


class ArtifactExistsError(ArtifactStoreError):
    """Raised when an immutable artifact or manifest would be overwritten."""


class ArtifactIntegrityError(ArtifactStoreError):
    """Raised when an existing content-addressed file does not match its hash."""


class ArtifactNotFoundError(ArtifactStoreError):
    """Raised when an artifact or manifest is unavailable."""


class ArtifactPathError(ArtifactStoreError, ValueError):
    """Raised when a path is outside the artifact store's safe scope."""


def sha256_bytes(data: bytes) -> str:
    """Return the lowercase SHA-256 digest for ``data``."""

    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path | str, *, chunk_size: int = _CHUNK_SIZE) -> str:
    """Hash a regular file without loading the complete file into memory."""

    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(file_path)
    digest = hashlib.sha256()
    with file_path.open("rb") as source:
        for chunk in iter(lambda: source.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class StoredArtifact:
    """An artifact manifest together with its immutable filesystem locations."""

    manifest: ArtifactManifest | SourceArtifact | DerivedArtifact
    content_path: Path
    manifest_path: Path

    @property
    def artifact_id(self) -> str:
        return self.manifest.id

    @property
    def content_hash(self) -> str:
        return self.manifest.content_hash

    @property
    def immutable_location(self) -> str:
        return self.manifest.immutable_location

    @property
    def size_bytes(self) -> int | None:
        return self.manifest.size_bytes

    def registration_metadata(self) -> dict[str, object]:
        """Return JSON/SQLite-friendly metadata for a later ledger registration."""

        registration: dict[str, object] = {
            "id": self.manifest.id,
            "artifact_id": self.manifest.id,
            "kind": self.manifest.kind.value,
            "content_hash": self.manifest.content_hash,
            "media_type": self.manifest.media_type,
            "immutable_location": self.manifest.immutable_location,
            "manifest_location": f"manifests/{self.manifest.id}.json",
            "size_bytes": self.manifest.size_bytes,
            "parent_artifact_ids": list(self.manifest.parent_artifact_ids),
            "metadata": dict(self.manifest.metadata),
        }
        if isinstance(self.manifest, SourceArtifact):
            registration.update(
                {
                    "role": self.manifest.role.value,
                    "format": self.manifest.format,
                    "original_location": self.manifest.original_location,
                }
            )
        if isinstance(self.manifest, DerivedArtifact):
            registration.update(
                {
                    "transformation": self.manifest.transformation,
                    "fit_scope": self.manifest.fit_scope,
                }
            )
        return registration


class FileSystemArtifactStore:
    """Store immutable files under a content-addressed artifact root.

    Objects live at ``objects/<first-two-hash-characters>/<remaining-hash>`` and
    manifests live at ``manifests/<hash>.json``. The SHA-256 digest is also the
    stable artifact ID, which makes duplicate content resolve to one identity.
    """

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root).expanduser().resolve()
        self.objects_root = self.root / "objects"
        self.manifests_root = self.root / "manifests"
        self._temporary_root = self.root / ".tmp"
        for directory in (self.objects_root, self.manifests_root, self._temporary_root):
            directory.mkdir(parents=True, exist_ok=True)

    def put_source(
        self,
        source_path: Path | str,
        *,
        role: DatasetRole | str = DatasetRole.AUXILIARY,
        media_type: str = "application/octet-stream",
        format: str | None = None,
        original_location: str | None = None,
        metadata: dict[str, str] | None = None,
    ) -> StoredArtifact:
        """Copy a source file into immutable storage and write its manifest."""

        source = Path(source_path)
        if not source.is_file():
            raise FileNotFoundError(source)
        source_format = format or source.suffix.lstrip(".").lower() or "binary"
        source_role = DatasetRole(role)
        digest, size = self._stage_payload(lambda target: self._copy_file(source, target))
        existing = self._existing_artifact(digest)
        if existing is not None:
            return existing

        manifest = SourceArtifact(
            id=digest,
            role=source_role,
            content_hash=digest,
            media_type=media_type,
            immutable_location=self._object_location(digest),
            size_bytes=size,
            metadata=dict(metadata or {}),
            format=source_format,
            original_location=original_location if original_location is not None else str(source),
        )
        return self._finish_new_artifact(manifest)

    def put_derived(
        self,
        content: Path | str | bytes | bytearray | BinaryIO,
        *,
        parent_artifact_ids: Sequence[str],
        transformation: str,
        fit_scope: str | None = None,
        media_type: str = "application/octet-stream",
        metadata: dict[str, str] | None = None,
    ) -> StoredArtifact:
        """Store transformed content and record its parent and fit scope."""

        parents = self._validate_parent_ids(parent_artifact_ids)
        if not transformation.strip():
            raise ValueError("transformation must not be empty")
        if fit_scope is not None and not fit_scope.strip():
            raise ValueError("fit_scope must not be empty when provided")
        digest, size = self._stage_payload(lambda target: self._copy_input(content, target))
        existing = self._existing_artifact(digest)
        if existing is not None:
            return existing

        lineage = ArtifactLineage(
            parent_artifact_ids=parents,
            transformation=transformation,
            fit_scope=fit_scope,
        )
        manifest = DerivedArtifact(
            id=digest,
            content_hash=digest,
            media_type=media_type,
            immutable_location=self._object_location(digest),
            size_bytes=size,
            parent_artifact_ids=parents,
            transformation=transformation,
            fit_scope=fit_scope,
            lineage=lineage,
            metadata=dict(metadata or {}),
        )
        return self._finish_new_artifact(manifest)

    def put_content(
        self,
        content: Path | str | bytes | bytearray | BinaryIO,
        *,
        kind: ArtifactKind = ArtifactKind.OTHER,
        media_type: str = "application/octet-stream",
        metadata: dict[str, str] | None = None,
    ) -> StoredArtifact:
        """Store standalone immutable content without inventing a dataset parent."""

        digest, size = self._stage_payload(lambda target: self._copy_input(content, target))
        existing = self._existing_artifact(digest)
        if existing is not None:
            return existing
        manifest = ArtifactManifest(
            id=digest,
            kind=kind,
            content_hash=digest,
            media_type=media_type,
            immutable_location=self._object_location(digest),
            size_bytes=size,
            metadata=dict(metadata or {}),
        )
        return self._finish_new_artifact(manifest)

    store_source = put_source
    store_derived = put_derived
    store_content = put_content
    add_source = put_source
    add_derived = put_derived

    def get_manifest(self, artifact_id: str) -> ArtifactManifest | SourceArtifact | DerivedArtifact:
        """Read and validate one manifest without following an unsafe path."""

        digest = self._validate_artifact_id(artifact_id)
        path = self.manifest_path(digest)
        if not path.is_file():
            raise ArtifactNotFoundError(f"manifest not found for artifact {digest}")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ArtifactIntegrityError(f"invalid artifact manifest: {path}") from error
        if not isinstance(payload, dict):
            raise ArtifactIntegrityError("artifact manifest must contain a JSON object")
        if payload.get("id") != digest:
            raise ArtifactIntegrityError(f"manifest identity mismatch: {path}")
        manifest = self._parse_manifest(payload)
        self._validate_manifest_location(manifest)
        return manifest

    def read_bytes(self, artifact: str | ArtifactManifest | StoredArtifact) -> bytes:
        """Safely read an artifact and verify its content hash on readback."""

        manifest = self._manifest_for(artifact)
        path = self.artifact_path(manifest)
        if not path.is_file():
            raise ArtifactNotFoundError(f"content not found for artifact {manifest.id}")
        try:
            with self._open_nofollow(path) as source:
                data = source.read()
        except OSError as error:
            raise ArtifactNotFoundError(
                f"content not readable for artifact {manifest.id}"
            ) from error
        if sha256_bytes(data) != manifest.content_hash.lower():
            raise ArtifactIntegrityError(f"content hash mismatch for artifact {manifest.id}")
        return data

    def open_read(self, artifact: str | ArtifactManifest | StoredArtifact) -> BinaryIO:
        """Open an artifact for safe read-only access without trusting user paths."""

        manifest = self._manifest_for(artifact)
        path = self.artifact_path(manifest)
        if not path.is_file():
            raise ArtifactNotFoundError(f"content not found for artifact {manifest.id}")
        try:
            return self._open_nofollow(path)
        except OSError as error:
            raise ArtifactNotFoundError(
                f"content not readable for artifact {manifest.id}"
            ) from error

    def artifact_path(self, artifact: str | ArtifactManifest | StoredArtifact) -> Path:
        """Return the verified object path for an artifact or artifact ID."""

        manifest = self._manifest_for(artifact)
        self._validate_manifest_location(manifest)
        return self._safe_relative_path(manifest.immutable_location)

    def manifest_path(self, artifact_id: str) -> Path:
        """Return a safe manifest path for an artifact ID."""

        digest = self._validate_artifact_id(artifact_id)
        return self._safe_relative_path(f"manifests/{digest}.json")

    def resolve_location(self, relative_path: str) -> Path:
        """Resolve a store-relative location while rejecting traversal and links."""

        return self._safe_relative_path(relative_path)

    def manifest_json(self, artifact: str | ArtifactManifest | StoredArtifact) -> str:
        """Return the persisted deterministic manifest JSON."""

        manifest = self._manifest_for(artifact)
        return manifest.canonical_json() + "\n"

    def _finish_new_artifact(
        self, manifest: SourceArtifact | DerivedArtifact | ArtifactManifest
    ) -> StoredArtifact:
        content_path = self._safe_relative_path(manifest.immutable_location)
        manifest_path = self.manifest_path(manifest.id)
        if manifest_path.exists():
            existing = self.get_manifest(manifest.id)
            return StoredArtifact(existing, content_path, manifest_path)
        payload = (manifest.canonical_json() + "\n").encode("utf-8")
        try:
            with manifest_path.open("xb") as destination:
                destination.write(payload)
                destination.flush()
                os.fsync(destination.fileno())
        except FileExistsError:
            existing = self.get_manifest(manifest.id)
            return StoredArtifact(existing, content_path, manifest_path)
        return StoredArtifact(manifest, content_path, manifest_path)

    def _existing_artifact(self, digest: str) -> StoredArtifact | None:
        manifest_path = self.manifest_path(digest)
        if not manifest_path.exists():
            return None
        manifest = self.get_manifest(digest)
        content_path = self.artifact_path(manifest)
        return StoredArtifact(manifest, content_path, manifest_path)

    def _stage_payload(self, writer: Callable[[BinaryIO], int]) -> tuple[str, int]:
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w+b", dir=self._temporary_root, prefix="artifact-", delete=False
            ) as destination:
                temporary = Path(destination.name)
                size = writer(cast(BinaryIO, destination))
                destination.flush()
                os.fsync(destination.fileno())
                destination.seek(0)
                digest = hashlib.sha256()
                for chunk in iter(lambda: destination.read(_CHUNK_SIZE), b""):
                    digest.update(chunk)
            target = self._object_path(digest.hexdigest())
            target.parent.mkdir(parents=True, exist_ok=True)
            self._install_object(temporary, target, digest.hexdigest())
            temporary = None
            return digest.hexdigest(), size
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    @staticmethod
    def _copy_file(source: Path, destination: BinaryIO) -> int:
        size = 0
        with source.open("rb") as source_stream:
            for chunk in iter(lambda: source_stream.read(_CHUNK_SIZE), b""):
                destination.write(chunk)
                size += len(chunk)
        return size

    @staticmethod
    def _copy_input(
        content: Path | str | bytes | bytearray | BinaryIO, destination: BinaryIO
    ) -> int:
        if isinstance(content, (Path, str)):
            return FileSystemArtifactStore._copy_file(Path(content), destination)
        stream: BinaryIO
        close_stream = False
        if isinstance(content, (bytes, bytearray)):
            stream = io.BytesIO(bytes(content))
            close_stream = True
        else:
            stream = content
        size = 0
        try:
            for chunk in iter(lambda: stream.read(_CHUNK_SIZE), b""):
                if not isinstance(chunk, bytes):
                    raise TypeError("derived content streams must return bytes")
                destination.write(chunk)
                size += len(chunk)
        finally:
            if close_stream:
                stream.close()
        return size

    def _install_object(self, temporary: Path, target: Path, digest: str) -> None:
        if target.exists():
            if target.is_symlink() or not target.is_file():
                raise ArtifactIntegrityError(f"object path is not a regular file: {target}")
            if sha256_file(target) != digest:
                raise ArtifactIntegrityError(f"object hash mismatch at {target}")
            temporary.unlink(missing_ok=True)
            return
        try:
            os.link(temporary, target)
        except FileExistsError:
            if target.is_symlink() or not target.is_file() or sha256_file(target) != digest:
                raise ArtifactIntegrityError(f"object hash mismatch at {target}")
        finally:
            temporary.unlink(missing_ok=True)

    def _manifest_for(
        self, artifact: str | ArtifactManifest | StoredArtifact
    ) -> ArtifactManifest | SourceArtifact | DerivedArtifact:
        if isinstance(artifact, StoredArtifact):
            return artifact.manifest
        if isinstance(artifact, (ArtifactManifest, SourceArtifact, DerivedArtifact)):
            return artifact
        return self.get_manifest(artifact)

    def _object_path(self, digest: str) -> Path:
        return self._safe_relative_path(self._object_location(digest))

    @staticmethod
    def _object_location(digest: str) -> str:
        return f"objects/{digest[:2]}/{digest[2:]}"

    def _validate_manifest_location(self, manifest: ArtifactManifest) -> None:
        expected = self._object_location(manifest.content_hash.lower())
        if manifest.immutable_location != expected:
            raise ArtifactPathError("manifest points outside its content-addressed object path")

    @staticmethod
    def _parse_manifest(payload: object) -> ArtifactManifest | SourceArtifact | DerivedArtifact:
        if not isinstance(payload, dict):
            raise ArtifactIntegrityError("artifact manifest must contain a JSON object")
        try:
            if payload.get("kind") == ArtifactKind.DATASET.value and "role" in payload:
                return SourceArtifact.model_validate(payload)
            if payload.get("kind") == ArtifactKind.DERIVED_DATASET.value:
                return DerivedArtifact.model_validate(payload)
            return ArtifactManifest.model_validate(payload)
        except ValueError as error:
            raise ArtifactIntegrityError("artifact manifest failed validation") from error

    def _validate_artifact_id(self, artifact_id: str) -> str:
        if not isinstance(artifact_id, str) or not _SHA256_PATTERN.fullmatch(artifact_id):
            raise ArtifactPathError("artifact ID must be a SHA-256 hex digest")
        return artifact_id.lower()

    @staticmethod
    def _validate_parent_ids(parent_artifact_ids: Sequence[str]) -> list[StableId]:
        if isinstance(parent_artifact_ids, (str, bytes)) or not parent_artifact_ids:
            raise ValueError("parent_artifact_ids must contain at least one artifact ID")
        parents: list[StableId] = []
        for parent_id in parent_artifact_ids:
            parents.append(validate_identifier(str(parent_id), "parent artifact ID"))
        return parents

    def _safe_relative_path(self, relative_path: str) -> Path:
        if not relative_path or "\x00" in relative_path or "\\" in relative_path:
            raise ArtifactPathError("artifact paths must be relative POSIX paths")
        candidate = Path(relative_path)
        if candidate.is_absolute() or any(part in {"", ".", ".."} for part in candidate.parts):
            raise ArtifactPathError("artifact path escapes the artifact root")
        resolved = (self.root / candidate).resolve(strict=False)
        try:
            resolved.relative_to(self.root)
        except ValueError as error:
            raise ArtifactPathError("artifact path escapes the artifact root") from error
        return resolved

    @staticmethod
    def _open_nofollow(path: Path) -> BinaryIO:
        flags = os.O_RDONLY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(path, flags)
        return os.fdopen(descriptor, "rb")


FilesystemArtifactStore = FileSystemArtifactStore
ArtifactStore = FileSystemArtifactStore
