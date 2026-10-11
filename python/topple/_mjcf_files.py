"""The files an MJCF references, resolved the way MuJoCo resolves them (contract §3, §13).

``robot_hash`` and ``bundle_hash`` cover them: ``<include>`` (recursively), meshes, skins and
height fields (``meshdir``), textures (``texturedir``), sub-models with every file they
reference, and any other ``file`` attribute.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

_FILE_ATTRS = ("file", "fileright", "fileleft", "fileup", "filedown", "filefront", "fileback")
_MESH_DIR_TAGS = frozenset({"mesh", "skin", "hfield"})  # MuJoCo resolves these via meshdir
_SUBMODEL_TAG = "model"  # <asset><model file="..."/>: a whole MJCF that <attach> places


def _mjcf_trees(path: Path, model_dir: Path, seen: set[Path]) -> list[ET.Element]:
    """The root element of ``path`` and of every file it ``<include>``s, recursively."""
    root = ET.parse(path).getroot()
    trees = [root]
    for include in root.iter("include"):
        target = _resolve(model_dir, "", include.get("file", ""), path)
        if target not in seen:
            seen.add(target)
            trees += _mjcf_trees(target, model_dir, seen)
    return trees


def _resolve(model_dir: Path, subdir: str, name: str, owner: Path) -> Path:
    path = (model_dir / subdir / name).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{owner.name} references {name}, which does not exist ({path})")
    return path


def _compiler_dirs(trees: list[ET.Element]) -> tuple[str, str]:
    """(meshdir, texturedir) as MuJoCo applies them: ``assetdir`` sets both, specific ones win."""
    meshdir = texturedir = ""
    for tree in trees:
        for compiler in tree.iter("compiler"):
            assetdir = compiler.get("assetdir")
            meshdir = compiler.get("meshdir", assetdir if assetdir is not None else meshdir)
            texturedir = compiler.get(
                "texturedir", assetdir if assetdir is not None else texturedir
            )
    return meshdir, texturedir


def referenced_files(xml_path: Path) -> tuple[Path, ...]:
    """Every file an MJCF references, resolved the way MuJoCo resolves it, sorted.

    Covers ``<include>`` (recursively), meshes, skins and height fields (``meshdir``), textures
    including cube faces (``texturedir``), sub-models (``<asset><model file>``) together with every
    file *they* reference, and any other element with a ``file`` attribute (relative to the
    model's directory). A missing file raises ``FileNotFoundError``.
    """
    return tuple(sorted(_model_files(xml_path, frozenset({xml_path.resolve()}))))


def _asset_subdir(tag: str, meshdir: str, texturedir: str) -> str:
    """The compiler directory MuJoCo resolves an element's ``file`` against."""
    if tag in _MESH_DIR_TAGS:
        return meshdir
    return texturedir if tag == "texture" else ""


def _model_files(xml_path: Path, models: frozenset[Path]) -> frozenset[Path]:
    """The files the MJCF at ``xml_path`` references, sub-models' files included.

    MuJoCo loads a sub-model (``<asset><model file>``, found relative to this model's directory)
    as a model of its own: its files resolve against *its* directory and *its* ``<compiler>``, so
    it is walked the same way. ``models`` holds the sub-models on the current path, so two that
    name each other are each listed but walked once.
    """
    model_dir = xml_path.parent
    includes: set[Path] = set()
    trees = _mjcf_trees(xml_path, model_dir, includes)
    meshdir, texturedir = _compiler_dirs(trees)
    found = set(includes)
    for tree in trees:
        for element in tree.iter():
            if element.tag in ("include", "compiler"):
                continue
            subdir = _asset_subdir(element.tag, meshdir, texturedir)
            for name in filter(None, (element.get(attr) for attr in _FILE_ATTRS)):
                target = _resolve(model_dir, subdir, name, xml_path)
                found.add(target)
                if element.tag == _SUBMODEL_TAG and target not in models:
                    found |= _model_files(target, models | {target})
    return frozenset(found)
