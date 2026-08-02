from __future__ import annotations

import numpy as np
import trimesh

import isaaclab.terrains.trimesh.utils as mesh_utils_terrains
from isaaclab.terrains import SubTerrainBaseCfg
from isaaclab.utils import configclass


def maze_terrain(difficulty: float, cfg: MazeTerrainCfg) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Generate a grid maze with a randomized depth-first search.
    """
    rng = np.random.default_rng()
    num_cells = int(round(cfg.num_cells_range[0] + difficulty * (cfg.num_cells_range[1] - cfg.num_cells_range[0])))
    num_cells = max(num_cells, 3)
    cell_size_x = cfg.size[0] / num_cells
    cell_size_y = cfg.size[1] / num_cells

    # horiz_walls[gy, gx]: wall segment spanning x at y = gy * cell_size_y (separates row gy-1 from row gy)
    # vert_walls[gy, gx]: wall segment spanning y at x = gx * cell_size_x (separates column gx-1 from column gx)
    horiz_walls = np.ones((num_cells + 1, num_cells), dtype=bool)
    vert_walls = np.ones((num_cells, num_cells + 1), dtype=bool)

    # carve a perfect maze with a randomized depth-first search (recursive backtracker), starting from the
    # center cell so the robot's spawn point is guaranteed to reach every corridor.
    visited = np.zeros((num_cells, num_cells), dtype=bool)
    start_cell = (num_cells // 2, num_cells // 2)
    visited[start_cell] = True
    stack = [start_cell]
    while stack:
        cy, cx = stack[-1]
        neighbors = []
        if cy > 0 and not visited[cy - 1, cx]:
            neighbors.append((cy - 1, cx, "S"))
        if cy < num_cells - 1 and not visited[cy + 1, cx]:
            neighbors.append((cy + 1, cx, "N"))
        if cx > 0 and not visited[cy, cx - 1]:
            neighbors.append((cy, cx - 1, "W"))
        if cx < num_cells - 1 and not visited[cy, cx + 1]:
            neighbors.append((cy, cx + 1, "E"))
        if not neighbors:
            stack.pop()
            continue
        ny, nx, direction = neighbors[rng.integers(len(neighbors))]
        if direction == "N":
            horiz_walls[cy + 1, cx] = False
        elif direction == "S":
            horiz_walls[cy, cx] = False
        elif direction == "E":
            vert_walls[cy, cx + 1] = False
        elif direction == "W":
            vert_walls[cy, cx] = False
        visited[ny, nx] = True
        stack.append((ny, nx))

    meshes_list = []
    wt = cfg.wall_thickness
    wh = cfg.wall_height
    for gy in range(num_cells + 1):
        for gx in range(num_cells):
            if horiz_walls[gy, gx]:
                pos = ((gx + 0.5) * cell_size_x, gy * cell_size_y, 0.5 * wh)
                dim = (cell_size_x + wt, wt, wh)
                meshes_list.append(trimesh.creation.box(dim, trimesh.transformations.translation_matrix(pos)))
    for gy in range(num_cells):
        for gx in range(num_cells + 1):
            if vert_walls[gy, gx]:
                pos = (gx * cell_size_x, (gy + 0.5) * cell_size_y, 0.5 * wh)
                dim = (wt, cell_size_y + wt, wh)
                meshes_list.append(trimesh.creation.box(dim, trimesh.transformations.translation_matrix(pos)))

    ground = mesh_utils_terrains.make_plane(cfg.size, height=0.0, center_zero=False)
    meshes_list.append(ground)

    origin = np.asarray([(start_cell[1] + 0.5) * cell_size_x, (start_cell[0] + 0.5) * cell_size_y, 0.0])
    return meshes_list, origin


@configclass
class MazeTerrainCfg(SubTerrainBaseCfg):
    """Configuration for a procedurally generated grid-maze terrain."""

    function = maze_terrain

    num_cells_range: tuple[int, int] = (3, 5)
    """Min/max number of cells per side of the maze grid, interpolated by difficulty."""

    wall_height: float = 0.5
    """Height of the maze walls (in m)."""

    wall_thickness: float = 0.05
    """Thickness of the maze walls (in m)."""