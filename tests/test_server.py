import json

import pytest
from mcp.shared.memory import (
    create_connected_server_and_client_session as client_session,
)

from mesh_validation_mcp.server import mcp as server_app

pytestmark = pytest.mark.anyio


async def test_list_tools():
    async with client_session(server_app._mcp_server) as client:
        result = await client.list_tools()
        names = {tool.name for tool in result.tools}
        assert names == {
            "inspect_mesh",
            "validate_mesh",
            "render_mesh",
            "compare_meshes",
            "assert_properties",
            "compare_to_golden",
            "inspect_section",
            "measure_thickness",
            "analyze_draft",
            "fit_feature",
            "validate_boolean",
            "detect_symmetry",
            "measure_displacement",
            "validate_array",
            "validate_generative",
            "validate_remesh",
            "check_clearance",
            "compare_silhouette",
            "units_sanity",
            "validate_dfm",
        }
        validate = next(t for t in result.tools if t.name == "validate_mesh")
        schema = json.dumps(validate.inputSchema)
        assert "volume" in schema and "bbox_extents" in schema and "watertight" in schema


async def test_inspect_mesh(box_path):
    async with client_session(server_app._mcp_server) as client:
        result = await client.call_tool("inspect_mesh", {"file_path": box_path})
        assert not result.isError
        assert result.structuredContent["volume"] == pytest.approx(500.0, rel=1e-5)
        assert result.structuredContent["is_watertight"] is True


async def test_validate_pass_with_render(box_path):
    async with client_session(server_app._mcp_server) as client:
        result = await client.call_tool(
            "validate_mesh",
            {
                "file_path": box_path,
                "expectations": {"volume": 500, "watertight": True, "face_count": 12},
            },
        )
        assert not result.isError
        report = json.loads(result.content[0].text)
        assert report["passed"] is True
        assert report["render"]["included"] is True
        assert any(
            block.type == "image" and block.mimeType == "image/png"
            for block in result.content
        )


async def test_validate_fail_without_render(box_path):
    async with client_session(server_app._mcp_server) as client:
        result = await client.call_tool(
            "validate_mesh",
            {
                "file_path": box_path,
                "expectations": {"volume": 900},
                "include_render": False,
            },
        )
        assert not result.isError
        assert len(result.content) == 1
        report = json.loads(result.content[0].text)
        assert report["passed"] is False
        assert report["checks"][0]["deviation_pct"] == pytest.approx(-44.44, abs=0.05)
        assert report["render"]["included"] is False


async def test_validate_unknown_expectation_key(box_path):
    async with client_session(server_app._mcp_server) as client:
        result = await client.call_tool(
            "validate_mesh",
            {"file_path": box_path, "expectations": {"bounding_box": [1, 2, 3]}},
        )
        assert result.isError
        assert "INVALID_EXPECTATION" in result.content[0].text


async def test_file_not_found_error_envelope():
    async with client_session(server_app._mcp_server) as client:
        result = await client.call_tool(
            "validate_mesh",
            {"file_path": "/definitely/not/here.stl", "expectations": {"volume": 1}},
        )
        assert result.isError
        assert "FILE_NOT_FOUND" in result.content[0].text


async def test_render_mesh_tool(box_path):
    async with client_session(server_app._mcp_server) as client:
        result = await client.call_tool(
            "render_mesh", {"file_path": box_path, "views": ["iso", "top"]}
        )
        assert not result.isError
        meta = json.loads(result.content[0].text)
        assert meta["backend"] in ("matplotlib", "pyrender")
        assert sum(1 for b in result.content if b.type == "image") == 1


async def test_compare_meshes_tool(box_path, scaled_path):
    async with client_session(server_app._mcp_server) as client:
        result = await client.call_tool(
            "compare_meshes", {"file_a": box_path, "file_b": scaled_path}
        )
        assert not result.isError
        report = json.loads(result.content[0].text)
        assert report["classification"] == "similarity"
        assert report["transform"]["uniform_scale"] == pytest.approx(2.0, rel=1e-3)
        assert any(b.type == "image" for b in result.content)


async def test_inspect_reports_integrity(box_path):
    async with client_session(server_app._mcp_server) as client:
        result = await client.call_tool("inspect_mesh", {"file_path": box_path})
        assert result.structuredContent["integrity"]["self_intersecting_face_count"] == 0
        assert result.structuredContent["integrity"]["non_manifold_edge_count"] == 0


async def test_validate_integrity_fails_and_highlights(self_intersecting_path):
    async with client_session(server_app._mcp_server) as client:
        result = await client.call_tool(
            "validate_mesh",
            {
                "file_path": self_intersecting_path,
                "expectations": {"self_intersecting_face_count": 0},
            },
        )
        assert not result.isError
        report = json.loads(result.content[0].text)
        assert report["passed"] is False
        assert report["render"]["defects_highlighted"] is True
        assert any(b.type == "image" for b in result.content)


async def test_compare_localized_roi(plate_path, emboss_good_path):
    async with client_session(server_app._mcp_server) as client:
        result = await client.call_tool(
            "compare_meshes",
            {
                "file_a": plate_path,
                "file_b": emboss_good_path,
                "localized": {
                    "region": {"kind": "box", "min": [-12, -12, 1.5], "max": [12, 12, 8]},
                    "emboss_height": 3.0,
                    "max_unchanged_deviation": 0.01,
                },
            },
        )
        assert not result.isError
        report = json.loads(result.content[0].text)
        assert report["localized"]["passed"] is True
        assert report["localized"]["stats"]["signed_displacement"]["peak"] == pytest.approx(
            3.0, abs=1e-3
        )


async def test_compare_localized_detects_leak(plate_path, emboss_bad_path):
    async with client_session(server_app._mcp_server) as client:
        result = await client.call_tool(
            "compare_meshes",
            {
                "file_a": plate_path,
                "file_b": emboss_bad_path,
                "localized": {
                    "region": {"kind": "box", "min": [-12, -12, 1.5], "max": [12, 12, 8]},
                    "max_unchanged_deviation": 0.01,
                },
            },
        )
        report = json.loads(result.content[0].text)
        assert report["localized"]["passed"] is False


async def test_invalid_region_error(box_path):
    async with client_session(server_app._mcp_server) as client:
        result = await client.call_tool(
            "compare_meshes",
            {
                "file_a": box_path,
                "file_b": box_path,
                "localized": {"region": {"kind": "vertex_ids", "vertex_ids": [999999]}},
            },
        )
        assert result.isError
        assert "INVALID_REGION" in result.content[0].text
