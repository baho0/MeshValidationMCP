import pytest

from mesh_validation_mcp.errors import ErrorCode, MeshToolError
from mesh_validation_mcp.loading import load_mesh


def test_load_box_stl(box_path):
    loaded = load_mesh(box_path)
    assert loaded.format == "stl"
    assert len(loaded.combined.faces) == 12
    assert len(loaded.combined.vertices) == 8
    assert len(loaded.bodies) == 1
    assert loaded.file_size_bytes > 0


def test_relative_path_rejected():
    with pytest.raises(MeshToolError) as exc:
        load_mesh("box.stl")
    assert exc.value.code is ErrorCode.NOT_ABSOLUTE_PATH


def test_missing_file():
    with pytest.raises(MeshToolError) as exc:
        load_mesh("/definitely/not/here.stl")
    assert exc.value.code is ErrorCode.FILE_NOT_FOUND


def test_unsupported_format(tmp_path):
    path = tmp_path / "notes.txt"
    path.write_text("not a mesh")
    with pytest.raises(MeshToolError) as exc:
        load_mesh(str(path))
    assert exc.value.code is ErrorCode.UNSUPPORTED_FORMAT
    assert "stl" in (exc.value.hint or "")


def test_corrupt_file(tmp_path):
    path = tmp_path / "corrupt.stl"
    path.write_bytes(b"solid junk\nnot really an stl")
    with pytest.raises(MeshToolError) as exc:
        load_mesh(str(path))
    assert exc.value.code in (ErrorCode.LOAD_FAILED, ErrorCode.EMPTY_MESH)


def test_glb_scene_normalized(tmp_path, box):
    path = tmp_path / "box.glb"
    box.export(path)
    loaded = load_mesh(str(path))
    assert len(loaded.combined.faces) == 12
    assert len(loaded.bodies) == 1
    assert loaded.combined.volume == pytest.approx(500.0, rel=1e-5)


def test_two_bodies_split(two_bodies_path):
    loaded = load_mesh(two_bodies_path)
    assert len(loaded.bodies) == 2


def test_error_string_is_json_envelope():
    try:
        load_mesh("/definitely/not/here.stl")
    except MeshToolError as exc:
        assert '"code":' in str(exc) or '"code"' in str(exc)
        assert "FILE_NOT_FOUND" in str(exc)
