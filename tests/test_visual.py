"""Phase 7 — visual validation: every validator returns an inspectable render, and the key
ones paint the validated quantity onto the mesh (signed displacement, thickness, draft, ...)."""

import json

import numpy as np
import pytest
import trimesh
from mcp.shared.memory import (
    create_connected_server_and_client_session as client_session,
)

from mesh_validation_mcp.loading import load_mesh
from mesh_validation_mcp.rendering import scalars_to_face_colors
from mesh_validation_mcp.server import mcp as server_app

pytestmark = pytest.mark.anyio


def _images(result):
    return [b for b in result.content if b.type == "image"]


def _report(result):
    return json.loads(result.content[0].text)


# --- rendering helper unit tests ---


def test_symmetric_scalars_center_on_zero(box_path):
    mesh = load_mesh(box_path).combined
    scalars = np.linspace(-3.0, 1.0, len(mesh.vertices))  # asymmetric range
    _colors, cb = scalars_to_face_colors(mesh, scalars, "x", cmap="RdBu_r", symmetric=True)
    assert cb["vmin"] == pytest.approx(-cb["vmax"])  # zero sits at the colormap centre


def test_per_face_scalars_accept_face_length(box_path):
    mesh = load_mesh(box_path).combined
    face_scalars = np.arange(len(mesh.faces), dtype=float)
    colors, _cb = scalars_to_face_colors(mesh, face_scalars, "x", per_face=True)
    assert len(colors) == len(mesh.faces)


# --- tools now return an annotated render ---


async def test_measure_displacement_returns_signed_heatmap(plate_path, emboss_good_path):
    async with client_session(server_app._mcp_server) as client:
        result = await client.call_tool(
            "measure_displacement", {"file_a": plate_path, "file_b": emboss_good_path}
        )
        assert not result.isError
        report = _report(result)
        assert report["signed_peak"] > 0  # emboss added material outward
        assert report["render"]["included"] is True
        assert len(_images(result)) == 1


async def test_measure_displacement_render_toggle(plate_path, emboss_good_path):
    async with client_session(server_app._mcp_server) as client:
        result = await client.call_tool(
            "measure_displacement",
            {"file_a": plate_path, "file_b": emboss_good_path, "include_render": False},
        )
        assert _report(result)["render"]["included"] is False
        assert len(_images(result)) == 0


async def test_measure_thickness_returns_render(shell_path):
    async with client_session(server_app._mcp_server) as client:
        result = await client.call_tool("measure_thickness", {"file_path": shell_path})
        assert not result.isError
        report = _report(result)
        assert report["median_thickness"] == pytest.approx(2.0, abs=0.1)
        assert len(_images(result)) == 1


async def test_analyze_draft_returns_render(box_path):
    async with client_session(server_app._mcp_server) as client:
        result = await client.call_tool(
            "analyze_draft", {"file_path": box_path, "pull_direction": [0, 0, 1]}
        )
        report = _report(result)
        assert report["undercut_face_count"] == 2
        assert len(_images(result)) == 1


async def test_inspect_section_returns_render(box_path):
    async with client_session(server_app._mcp_server) as client:
        result = await client.call_tool(
            "inspect_section",
            {"file_path": box_path, "plane_origin": [0, 0, 0], "plane_normal": [0, 0, 1]},
        )
        report = _report(result)
        assert report["net_area"] == pytest.approx(100.0, abs=1e-3)
        assert len(_images(result)) == 1


async def test_fit_feature_returns_render(sphere_path):
    async with client_session(server_app._mcp_server) as client:
        result = await client.call_tool(
            "fit_feature",
            {
                "file_path": sphere_path,
                "region": {"kind": "box", "min": [-9, -9, -9], "max": [9, 9, 9]},
                "kind": "sphere",
            },
        )
        report = _report(result)
        assert report["radius"] == pytest.approx(2.0, rel=1e-3)
        assert len(_images(result)) == 1


async def test_validate_generative_returns_render(box_path):
    async with client_session(server_app._mcp_server) as client:
        result = await client.call_tool(
            "validate_generative",
            {"file_path": box_path, "operation": "extrude", "profile_area": 100.0, "height": 5.0},
        )
        report = _report(result)
        assert report["passed"] is True
        assert len(_images(result)) == 1


async def test_detect_symmetry_returns_render(box_path):
    async with client_session(server_app._mcp_server) as client:
        result = await client.call_tool("detect_symmetry", {"file_path": box_path})
        report = _report(result)
        assert len(report["mirror_planes"]) == 3
        assert len(_images(result)) == 1
