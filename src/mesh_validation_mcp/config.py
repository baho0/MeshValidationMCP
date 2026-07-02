"""Central configuration: caps, seeds and render defaults.

Values are deliberately conservative so a single stdio MCP request stays
bounded; the ``MESH_MCP_*`` environment variables allow per-installation
overrides without code changes.
"""

from __future__ import annotations

import os

# Single seed used for every random sampling operation -> deterministic output.
SEED = 0

MAX_FILE_MB = float(os.environ.get("MESH_MCP_MAX_FILE_MB", "500"))
MAX_FACES_LOAD = int(os.environ.get("MESH_MCP_MAX_FACES", "5000000"))

# Renderer selection: "matplotlib" | "pyrender" (unset -> auto-probe).
RENDERER_ENV = "MESH_MCP_RENDERER"
# Above this face count the render uses a seeded face subset (display only).
RENDER_MAX_FACES = int(os.environ.get("MESH_MCP_RENDER_MAX_FACES", "120000"))
# shaded_edges draws edge strokes only below this face count.
EDGE_MAX_FACES = 20_000
# Heatmap distances computed on a seeded vertex subset above this count.
HEATMAP_MAX_VERTICES = 100_000

DEFAULT_RESOLUTION = 1024
# validate_mesh ships JSON + image together; slightly smaller sheet saves tokens.
VALIDATE_RESOLUTION = 896
MIN_RESOLUTION = 256
MAX_RESOLUTION = 1280
MAX_COMBINED_VIEWS = 6

MAX_COMPARE_SAMPLES = 20_000

# --- Localized-change / integrity thresholds ---
# A vertex counts as "changed" when its displacement exceeds this fraction of the
# before-mesh bbox diagonal (also the default "rest is untouched" tolerance).
LOCALIZED_CHANGE_REL = 1e-4
# Triangle quality Q = 4*sqrt(3)*area / sum(edge^2); equilateral=1, sliver->0.
SLIVER_QUALITY = 0.05
# Broad-phase self-intersection is skipped above this face count (too slow for stdio).
SELF_INTERSECTION_MAX_FACES = int(
    os.environ.get("MESH_MCP_SELFINT_MAX_FACES", "200000")
)
