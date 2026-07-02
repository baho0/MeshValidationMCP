import io

import numpy as np
import pytest
from PIL import Image as PILImage

from mesh_validation_mcp.errors import ErrorCode, MeshToolError
from mesh_validation_mcp.loading import load_mesh
from mesh_validation_mcp.rendering import (
    body_face_colors,
    render_views,
    scalars_to_face_colors,
)

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def _decode(png: bytes) -> PILImage.Image:
    assert png.startswith(PNG_MAGIC)
    return PILImage.open(io.BytesIO(png))


def test_contact_sheet(box):
    images, meta = render_views(box, ["iso", "front", "top", "right"])
    assert len(images) == 1
    sheet = _decode(images[0])
    assert sheet.width > 400 and sheet.height > 400
    assert meta["backend"] in ("matplotlib", "pyrender")
    assert meta["views"] == ["iso", "front", "top", "right"]
    assert meta["render_decimated"] is False


def test_individual_images(box):
    images, meta = render_views(box, ["iso", "front"], combine=False)
    assert len(images) == 2
    for png in images:
        _decode(png)
    assert meta["combined"] is False


@pytest.mark.parametrize("style", ["shaded", "shaded_edges", "wireframe"])
def test_styles(box, style):
    images, _ = render_views(box, ["iso"], style=style)
    _decode(images[0])


def test_invalid_view(box):
    with pytest.raises(MeshToolError) as exc:
        render_views(box, ["diagonal"])
    assert exc.value.code is ErrorCode.INVALID_VIEW
    assert "iso" in (exc.value.hint or "")


def test_too_many_combined_views(box):
    with pytest.raises(MeshToolError) as exc:
        render_views(box, ["iso", "iso_back", "front", "back", "left", "right", "top"])
    assert exc.value.code is ErrorCode.INVALID_VIEW


def test_resolution_clamped(box):
    images, meta = render_views(box, ["iso"], resolution=999999)
    sheet = _decode(images[0])
    assert max(sheet.size) <= 1280
    assert meta["resolution"] == 1280


def test_heatmap_render_gradient(sphere):
    scalars = np.asarray(sphere.vertices)[:, 2]  # non-uniform
    colors, colorbar = scalars_to_face_colors(sphere, scalars, label="test distance")
    assert colors.shape == (len(sphere.faces), 3)
    assert colorbar["vmax"] > colorbar["vmin"]
    images, _ = render_views(sphere, ["iso", "front"], face_colors=colors, colorbar=colorbar)
    _decode(images[0])


def test_heatmap_render_uniform_scalars(box):
    # A box's 8 corners are equidistant from the origin: vmin == vmax must not crash
    # and must take the solid-colorbar branch.
    scalars = np.linalg.norm(np.asarray(box.vertices), axis=1)
    colors, colorbar = scalars_to_face_colors(box, scalars, label="test distance")
    assert colorbar["vmin"] == colorbar["vmax"]
    images, _ = render_views(box, ["iso"], face_colors=colors, colorbar=colorbar)
    _decode(images[0])


def test_body_tint(two_bodies_path):
    mesh = load_mesh(two_bodies_path).combined
    colors = body_face_colors(mesh)
    assert colors is not None
    assert colors.shape == (len(mesh.faces), 3)
    # two distinct tints
    assert len(np.unique(colors.round(3), axis=0)) == 2


def test_single_body_no_tint(box):
    assert body_face_colors(box) is None


def test_matplotlib_backend_directly(box):
    """The CPU fallback must keep working even on machines where the GL
    backend is available and auto-selected."""
    from mesh_validation_mcp.rendering.base import VIEWS, RenderRequest, RenderStyle
    from mesh_validation_mcp.rendering.compose import compose_sheet, encode_png
    from mesh_validation_mcp.rendering.matplotlib_backend import MatplotlibRenderer

    tiles = MatplotlibRenderer().render(
        RenderRequest(
            mesh=box,
            views=[VIEWS["iso"], VIEWS["front"]],
            tile_px=256,
            style=RenderStyle.SHADED_EDGES,
        )
    )
    assert len(tiles) == 2
    png = encode_png(compose_sheet(tiles))
    image = _decode(png)
    assert image.width > 400


def test_pyrender_backend_directly(box):
    from mesh_validation_mcp.rendering.base import VIEWS, RenderRequest, RenderStyle
    from mesh_validation_mcp.rendering.pyrender_backend import PyrenderRenderer

    if not PyrenderRenderer.available():
        pytest.skip("pyrender/EGL stack not available")
    tiles = PyrenderRenderer().render(
        RenderRequest(
            mesh=box, views=[VIEWS["iso"]], tile_px=256, style=RenderStyle.SHADED
        )
    )
    assert len(tiles) == 1
    assert tiles[0].image.size == (256, 256)
