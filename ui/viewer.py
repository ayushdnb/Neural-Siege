from __future__ import annotations

import os
import collections
from typing import List, Tuple, Optional, Dict, Any
import math
from pathlib import Path  # added for checkpoint path handling

import pygame
import torch
import torch.nn as nn
import numpy as np

import config
from .camera import Camera
from engine.agent_registry import (
    COL_ALIVE, COL_TEAM, COL_HP, COL_X, COL_Y, COL_UNIT,
    COL_HP_MAX, COL_VISION, COL_ATK, COL_AGENT_ID
)

# --- Constants & Configuration ---
FONT_NAME = "consolas"

# Palette
COLORS = {
    "bg": (20, 22, 28), "hud_bg": (12, 14, 18), "side_bg": (18, 20, 26),
    "grid": (40, 42, 48), "border": (70, 74, 82), "wall": (90, 94, 102),
    "empty": (24, 26, 32),

    # Red Team
    "red_soldier": (231, 76, 60),
    "red_archer":  (211, 84, 0),
    "red":         (231, 76, 60),

    # Blue Team
    "blue_soldier": (52, 152, 219),
    "blue_archer":  (22, 160, 133),
    "blue":         (52, 152, 219),

    "archer_glyph": (245, 230, 90), "marker": (242, 228, 92),
    "text": (230, 230, 230), "text_dim": (180, 186, 194),
    "green": (46, 204, 113), "warn": (243, 156, 18),
    "bar_bg": (38, 42, 48), "bar_fg": (46, 204, 113),
    "graph_red": (231, 76, 60, 150), "graph_blue": (52, 152, 219, 150),
    "graph_grid": (60, 60, 70), "pause_text": (241, 196, 15)
}

OVERLAYS = {
    "heal": (46, 204, 113, 60), "cp": (210, 210, 230, 48),
    "outline_red": (231, 76, 60, 160), "outline_blue": (52, 152, 219, 160),
    "outline_neutral": (160, 160, 170, 120),
    "threat_enemy": (231, 76, 60), "threat_ally": (52, 152, 219),
    "vision_range": (180, 180, 180, 40)
}
def _clamp_u8(x) -> int:
    """Clamp to [0,255] for pygame colors (RGBA)."""
    try:
        xi = int(x)
    except Exception:
        xi = 0
    if xi < 0:   return 0
    if xi > 255: return 255
    return xi

def _rgb(col) -> tuple[int, int, int]:
    """Return a safe (r,g,b) triple from a tuple/list of >=3 items."""
    r, g, b = (0, 0, 0)
    try:
        if len(col) >= 3:
            r, g, b = col[0], col[1], col[2]
    except Exception:
        pass
    return (_clamp_u8(r), _clamp_u8(g), _clamp_u8(b))
RAY_COLORS = {
    "ally": (52, 152, 219), "enemy": (231, 76, 60),
    "wall": (180, 180, 180), "empty": (100, 100, 110)
}


# =========================
# Utility
# =========================
def _center_window():
    os.environ.setdefault("SDL_VIDEO_CENTERED", "1")


def _get_model_summary(model: nn.Module) -> str:
    name = model.__class__.__name__.lower()
    if "transformer" in name:
        try:
            d = model.embed_dim
            num_cross = 1 if hasattr(model, "cross_attention") else 0
            num_self = 1 if hasattr(model, "self_attention") else 0
            return f"Transformer(d={d}, Cross={num_cross}, Self={num_self})"
        except Exception:
            return "TransformerBrain"
    try:
        linears = [m for m in model.modules() if isinstance(m, nn.Linear)]
        dims = [linears[0].in_features] + [m.out_features for m in linears]
        return "→".join(map(str, dims))
    except Exception:
        return "Unknown"


def _param_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class TextCache:
    """
    Caches:
      1) pygame Font objects per size  (self.fonts[size])
      2) rendered text surfaces per (text, size, color, aa)  (self.cache[key])

    Why your crash happened:
      - You only pre-created fonts for 13/16/18
      - But brain labels request size 10..16
      - So self.fonts[10] raised KeyError

    Fix A:
      - If a requested font size doesn't exist, create it on-demand.
    """
    def __init__(self):
        # Keep a few common defaults (fast warm-start)
        self.fonts = {
            13: self._mk_font(13),
            16: self._mk_font(16),
            18: self._mk_font(18),
        }

        # Rendered surface cache: (text, size, color, aa) -> pygame.Surface
        self.cache = {}

    def _mk_font(self, sz: int):
        """
        Build a pygame Font safely.
        SysFont can fail on some machines; fall back to default pygame font.
        """
        try:
            return pygame.font.SysFont(FONT_NAME, sz)
        except Exception:
            return pygame.font.Font(None, sz)

    def render(self, text: str, size: int, color, aa: bool = True):
        """
        Return a cached rendered surface for the given text settings.
        Creates the font for `size` if it wasn't pre-created (Fix A).
        """
        # --- Safety: normalize inputs (prevents weird cache blowups) ---
        if not isinstance(text, str):
            text = str(text)

        # Avoid invalid sizes; pygame fonts need positive ints.
        size = int(size)
        if size < 1:
            size = 1

        # --- Fix A: create missing font sizes on-demand ---
        if size not in self.fonts:
            self.fonts[size] = self._mk_font(size)

        key = (text, size, color, aa)

        # Render once per unique key
        if key not in self.cache:
            self.cache[key] = self.fonts[size].render(text, aa, color)

        return self.cache[key]



# =========================
# Layout
# =========================
class LayoutManager:
    def __init__(self, viewer): self.viewer = viewer
    def side_width(self): return max(320, min(420, int(self.viewer.Wpix * 0.27)))
    def world_rect(self):
        m = self.viewer.margin
        return pygame.Rect(m, m,
                           max(64, self.viewer.Wpix - self.side_width() - 3*m),
                           max(64, self.viewer.Hpix - 126 - 2*m))
    def side_rect(self):
        m, side_w = self.viewer.margin, self.side_width()
        return pygame.Rect(self.viewer.Wpix - side_w - m, m, side_w,
                           max(64, self.viewer.Hpix - 126 - 2*m))
    def hud_rect(self): return pygame.Rect(0, self.viewer.Hpix - 126, self.viewer.Wpix, 126)


# =========================
# World renderer
# =========================
class WorldRenderer:
    def __init__(self, viewer, grid, registry):
        self.viewer, self.grid, self.registry = viewer, grid, registry
        self.cam = viewer.cam
        self.static_surf = None
        self._zone_cache = {"heal_tiles": [], "cp_rects": []}

    def build_static_cache(self, engine):
        self.static_surf = None
        self._zone_cache = {"heal_tiles": [], "cp_rects": []}
        zones = getattr(engine, "zones", None)
        if zones:
            if getattr(zones, "heal_mask", None) is not None:
                ys, xs = torch.nonzero(zones.heal_mask, as_tuple=True)
                self._zone_cache["heal_tiles"] = list(zip(xs.cpu().tolist(), ys.cpu().tolist()))
            for m in getattr(zones, "cp_masks", []):
                ys, xs = torch.nonzero(m, as_tuple=True)
                if xs.numel() > 0:
                    self._zone_cache["cp_rects"].append((
                        xs.min().item(), ys.min().item(),
                        xs.max().item() + 1, ys.max().item() + 1
                    ))

    def _draw_static_background(self):
        wrect = self.viewer.layout.world_rect()
        self.static_surf = pygame.Surface(wrect.size)
        self.static_surf.fill(COLORS["bg"])

        H, W = self.grid.shape[1], self.grid.shape[2]
        occ_np = self.grid[0].detach().cpu().numpy()

        # Static walls/empty
        for y in range(H):
            for x in range(W):
                occ = occ_np[y, x]
                if occ in {0, 1}:
                    color = COLORS["empty"] if occ == 0 else COLORS["wall"]
                    cx, cy = self.cam.world_to_screen(x, y)
                    pygame.draw.rect(self.static_surf, color, (cx, cy, self.cam.cell_px, self.cam.cell_px))

        # Zones overlays
        overlay = pygame.Surface(wrect.size, pygame.SRCALPHA)
        for x, y in self._zone_cache["heal_tiles"]:
            cx, cy = self.cam.world_to_screen(x, y)
            pygame.draw.rect(overlay, OVERLAYS["heal"], (cx, cy, self.cam.cell_px, self.cam.cell_px))
        self.static_surf.blit(overlay, (0, 0))

    def draw(self, surf, state_data):
        wrect = self.viewer.layout.world_rect()
        if self.static_surf is None or self.static_surf.get_size() != wrect.size:
            self._draw_static_background()

        surf.blit(self.static_surf, wrect.topleft)

        c = self.cam.cell_px

        # CP overlay (cheap-ish)
        cp_overlay = pygame.Surface(wrect.size, pygame.SRCALPHA)
        for x0, y0, x1, y1 in self._zone_cache["cp_rects"]:
            patch = state_data["occ_np"][y0:y1, x0:x1]
            red_cnt, blue_cnt = (patch == 2).sum(), (patch == 3).sum()
            if red_cnt > blue_cnt:
                b_col, label = OVERLAYS["outline_red"], ("R", COLORS["red"])
            elif blue_cnt > red_cnt:
                b_col, label = OVERLAYS["outline_blue"], ("B", COLORS["blue"])
            else:
                b_col, label = OVERLAYS["outline_neutral"], ("–", COLORS["text_dim"])
            cx0, cy0 = self.cam.world_to_screen(x0, y0)
            cx1, cy1 = self.cam.world_to_screen(x1, y1)
            rect = pygame.Rect(cx0, cy0, cx1 - cx0, cy1 - cy0)
            pygame.draw.rect(cp_overlay, b_col, rect, max(1, c // 2))
            lab_surf = self.viewer.text_cache.render(label[0], 13, label[1])
            cp_overlay.blit(lab_surf, lab_surf.get_rect(center=rect.center))
        surf.blit(cp_overlay, wrect.topleft)

        # Agents
        for slot_id in state_data["alive_indices"]:
            entry = state_data["agent_map"].get(slot_id, None)
            if entry is None:
                continue
            x, y, unit, team, _uid, _btype = entry
            color_key = f"{'red' if team == 2.0 else 'blue'}_{'archer' if unit == 2.0 else 'soldier'}"
            color = COLORS[color_key]
            cx, cy = self.cam.world_to_screen(int(x), int(y))
            pygame.draw.rect(surf, color, (wrect.x + cx, wrect.y + cy, c, c))
            if unit == 2.0 and c > 4:
                pygame.draw.circle(
                    surf, COLORS["archer_glyph"],
                    (wrect.x + cx + c // 2, wrect.y + cy + c // 2),
                    max(2, c // 2 - 1),
                    max(1, c // 6)
                )

        if self.viewer.battle_view_enabled:
            self._draw_hp_bars(surf, wrect, c, state_data)
        if self.viewer.show_brain_types:
            self._draw_brain_labels(surf, wrect, c, state_data)
        if self.viewer.threat_vision_mode:
            self._draw_threat_vision(surf, wrect, c, state_data)
        if self.viewer.show_grid and c >= 6:
            self._draw_grid_lines(surf, wrect, c)
        self._draw_markers(surf, wrect, c, state_data["id_np"])
        if self.viewer.show_rays:
            self._draw_rays(surf, wrect, c, state_data)

    def _draw_hp_bars(self, surf, wrect, c, state_data):
        if c < 8:
            return
        hp_bar_surf = pygame.Surface(wrect.size, pygame.SRCALPHA)
        for slot_id in state_data["alive_indices"]:
            entry = state_data["agent_map"].get(slot_id, None)
            if entry is None:
                continue
            x, y, _unit, _team, _uid, _btype = entry

            hp_max = float(self.registry.agent_data[slot_id, COL_HP_MAX].item())
            hp = float(self.registry.agent_data[slot_id, COL_HP].item())
            hp_ratio_raw = (hp / hp_max) if hp_max > 0 else 0.0
            # Clamp ratio so alpha never goes out of [0..255]
            hp_ratio = max(0.0, min(1.0, hp_ratio_raw))
            cx, cy = self.cam.world_to_screen(int(x), int(y))
            bar_w, bar_h = c, max(1, c // 8)
            bar_y = cy - bar_h - 2
            if 0 <= bar_y < wrect.height and 0 <= cx < wrect.width:
                pygame.draw.rect(hp_bar_surf, (50, 50, 50), (cx, bar_y, bar_w, bar_h))
                pygame.draw.rect(hp_bar_surf, COLORS["green"], (cx, bar_y, bar_w * hp_ratio, bar_h))
        surf.blit(hp_bar_surf, wrect.topleft)

    def _draw_brain_labels(self, surf, wrect, c, state_data):
        """Renders the architecture type character right on top of the agent cell."""
        if c < 8:
            return
        for slot_id in state_data["alive_indices"]:
            entry = state_data["agent_map"].get(slot_id, None)
            if entry is None:
                continue
            x, y, _unit, _team, _uid, btype = entry

            cx, cy = self.cam.world_to_screen(int(x), int(y))
            if 0 <= cx < wrect.width and 0 <= cy < wrect.height:
                # Calculate size to fit dynamically inside the cell
                font_size = max(10, min(16, c // 2))
                lab_surf = self.viewer.text_cache.render(btype, font_size, COLORS["text"])
                rect = lab_surf.get_rect(center=(wrect.x + cx + c // 2, wrect.y + cy + c // 2))
                surf.blit(lab_surf, rect)

    def _draw_threat_vision(self, surf, wrect, c, state_data):
        slot_id = self.viewer.selected_slot_id
        if slot_id is None or slot_id not in state_data["agent_map"]:
            return

        my_x, my_y, _unit, my_team, _uid, _btype = state_data["agent_map"][slot_id]
        my_x = float(my_x); my_y = float(my_y)

        my_cx, my_cy = self.cam.world_to_screen(int(my_x), int(my_y))
        vision_range = float(self.registry.agent_data[slot_id, COL_VISION].item())
        vision_px = int(vision_range * c)

        overlay = pygame.Surface(wrect.size, pygame.SRCALPHA)
        center_px = (my_cx + c // 2, my_cy + c // 2)
        pygame.draw.circle(overlay, OVERLAYS["vision_range"], center_px, vision_px)

        for other_id, (ox, oy, _ou, o_team, _ouid, _obtype) in state_data["agent_map"].items():
            if slot_id == other_id:
                continue
            dist = float(np.hypot(ox - my_x, oy - my_y))
            if dist <= vision_range:
                o_cx, o_cy = self.cam.world_to_screen(int(ox), int(oy))
                hp_max = float(self.registry.agent_data[other_id, COL_HP_MAX].item())
                hp = float(self.registry.agent_data[other_id, COL_HP].item())
                hp_ratio = (hp / hp_max) if hp_max > 0 else 0.0
                if o_team != my_team:
                    radius = int(c * 0.7 * (0.5 + hp_ratio * 0.5))
                    alpha = _clamp_u8(50 + 150 * hp_ratio)
                    r, g, b = _rgb(OVERLAYS["threat_enemy"])
                    pygame.draw.circle(
                        overlay, (r, g, b, alpha),
                        (o_cx + c // 2, o_cy + c // 2), radius
                    )
                else:
                    r, g, b = _rgb(OVERLAYS["threat_ally"])
                    pygame.draw.circle(
                        overlay, (r, g, b, 100),
                        (o_cx + c // 2, o_cy + c // 2), int(c * 0.4), 1
                    )

        surf.blit(overlay, wrect.topleft)

    def _draw_grid_lines(self, surf, wrect, c):
        ax, ay = self.cam.world_to_screen(0, 0)
        off_x, off_y = (c - (ax % c)) % c, (c - (ay % c)) % c
        x, y = wrect.x + off_x, wrect.y + off_y
        while x < wrect.right:
            pygame.draw.line(surf, COLORS["grid"], (x, wrect.y), (x, wrect.bottom))
            x += c
        while y < wrect.bottom:
            pygame.draw.line(surf, COLORS["grid"], (wrect.x, y), (wrect.right, y))
            y += c
        pygame.draw.rect(surf, COLORS["border"], wrect, 2)

    def _draw_markers(self, surf, wrect, c, id_np):
        for slot_id in self.viewer.marked:
            pos = np.argwhere(id_np == slot_id)
            if pos.size > 0:
                y, x = pos[0]
                cx, cy = self.cam.world_to_screen(int(x), int(y))
                pygame.draw.rect(surf, COLORS["marker"], (wrect.x + cx, wrect.y + cy, c, c), max(1, c // 8))

    def _draw_rays(self, surf, wrect, c, state_data):
        slot_id = self.viewer.selected_slot_id
        if slot_id is None or slot_id not in state_data["agent_map"]:
            return

        agent_x, agent_y, _unit, my_team, _uid, _btype = state_data["agent_map"][slot_id]
        agent_x = float(agent_x); agent_y = float(agent_y)

        start_pos_screen = (
            wrect.x + self.cam.world_to_screen(int(agent_x), int(agent_y))[0] + c // 2,
            wrect.y + self.cam.world_to_screen(int(agent_x), int(agent_y))[1] + c // 2
        )

        vision_range = int(float(self.registry.agent_data[slot_id, COL_VISION].item()))
        occ_grid = state_data["occ_np"]
        H, W = occ_grid.shape

        num_rays_to_draw = 32
        for i in range(num_rays_to_draw):
            angle = i * (2 * math.pi / num_rays_to_draw)
            dx, dy = math.cos(angle), math.sin(angle)

            end_x, end_y = agent_x, agent_y
            color = RAY_COLORS["empty"]

            for step in range(1, vision_range + 1):
                test_x = int(round(agent_x + dx * step))
                test_y = int(round(agent_y + dy * step))

                if not (0 <= test_x < W and 0 <= test_y < H):
                    end_x, end_y = test_x, test_y
                    break

                occupant = occ_grid[test_y, test_x]
                if occupant != 0:
                    end_x, end_y = test_x, test_y
                    if occupant == 1:
                        color = RAY_COLORS["wall"]
                    else:
                        hit_team = occupant
                        color = RAY_COLORS["enemy"] if my_team != hit_team else RAY_COLORS["ally"]
                    break
            else:
                end_x = agent_x + dx * vision_range
                end_y = agent_y + dy * vision_range

            end_pos_world = self.cam.world_to_screen(int(end_x), int(end_y))
            end_pos_screen = (wrect.x + end_pos_world[0] + c // 2, wrect.y + end_pos_world[1] + c // 2)
            pygame.draw.line(surf, color, start_pos_screen, end_pos_screen, 1)


# =========================
# HUD / Side / Minimap
# =========================
class HudPanel:
    def __init__(self, viewer, stats):
        self.viewer, self.stats = viewer, stats
        self.score_history = collections.deque(maxlen=1000)

    def update(self):
        self.score_history.append(self.stats.red.score - self.stats.blue.score)

    def draw(self, surf, state_data):
        hud = self.viewer.layout.hud_rect()
        surf.fill(COLORS["hud_bg"], hud)
        y, x = hud.y + 8, 12

        pause_str = "[ PAUSED ]" if self.viewer.paused else f"[ {self.viewer.speed_multiplier}x ]"
        surf.blit(self.viewer.text_cache.render(
            f"Tick {self.stats.tick} {pause_str}", 16,
            COLORS["pause_text"] if self.viewer.paused else COLORS["text"]
        ), (x, y))

        self._draw_team_stats(surf, y, x, state_data)
        self._draw_score_graph(surf, hud)
        self.viewer.minimap.draw(surf, hud, state_data)

    def _draw_team_stats(self, surf, y, x, state_data):
        r_alive = sum(1 for _, _, _, team, _, _ in state_data["agent_map"].values() if team == 2.0)
        b_alive = len(state_data["agent_map"]) - r_alive
        rs_alive = sum(1 for _, _, unit, team, _, _ in state_data["agent_map"].values() if team == 2.0 and unit == 1.0)
        ra_alive = r_alive - rs_alive
        bs_alive = sum(1 for _, _, unit, team, _, _ in state_data["agent_map"].values() if team == 3.0 and unit == 1.0)
        ba_alive = b_alive - bs_alive

        red_str = f"Red  S:{self.stats.red.score:6.1f} CP:{self.stats.red.cp_points:4.1f} K:{self.stats.red.kills:3d} D:{self.stats.red.deaths:3d} Alive:{r_alive:3d} (S:{rs_alive} A:{ra_alive})"
        blue_str = f"Blue S:{self.stats.blue.score:6.1f} CP:{self.stats.blue.cp_points:4.1f} K:{self.stats.blue.kills:3d} D:{self.stats.blue.deaths:3d} Alive:{b_alive:3d} (S:{bs_alive} A:{ba_alive})"
        surf.blit(self.viewer.text_cache.render(red_str, 16, COLORS["red"]), (x, y + 24))
        surf.blit(self.viewer.text_cache.render(blue_str, 16, COLORS["blue"]), (x, y + 48))

    def _draw_score_graph(self, surf, hud):
        graph_rect = pygame.Rect(hud.right - 540, hud.y + 10, 300, hud.height - 20)
        pygame.draw.rect(surf, COLORS["bg"], graph_rect)
        if len(self.score_history) < 2:
            return
        scores = np.array(self.score_history, dtype=np.float32)
        max_abs_score = float(np.max(np.abs(scores)) or 1.0)
        norm_scores = scores / max_abs_score

        # pygame can choke on numpy scalar types; force plain Python numbers.
        denom = max(1, (len(scores) - 1))
        points = []
        for i, s in enumerate(norm_scores):
            s = float(s)  # convert np.float32 -> python float
            x = graph_rect.x + (i / denom) * graph_rect.width
            y = graph_rect.centery - s * (graph_rect.height / 2.2)

            # extra safety against NaN/Inf
            if not (math.isfinite(x) and math.isfinite(y)):
                continue

            # ints are safest for polygon/aalines in pygame
            points.append((int(x), int(y)))

        if len(points) < 2:
            return
        for i in range(1, 4):
            pygame.draw.line(
                surf, COLORS["graph_grid"],
                (graph_rect.x, graph_rect.y + i * graph_rect.height / 4),
                (graph_rect.right, graph_rect.y + i * graph_rect.height / 4),
            )

        red_poly_points = [(p[0], p[1]) for p in points if p[1] < graph_rect.centery]
        if red_poly_points:
            red_poly = [(graph_rect.x, graph_rect.centery)] + red_poly_points + [(red_poly_points[-1][0], graph_rect.centery)]
            pygame.draw.polygon(surf, COLORS["graph_red"], red_poly)

        blue_poly_points = [(p[0], p[1]) for p in points if p[1] > graph_rect.centery]
        if blue_poly_points:
            blue_poly = [(graph_rect.x, graph_rect.centery)] + blue_poly_points + [(blue_poly_points[-1][0], graph_rect.centery)]
            pygame.draw.polygon(surf, COLORS["graph_blue"], blue_poly)

        pygame.draw.aalines(surf, COLORS["text"], False, points)
        pygame.draw.rect(surf, COLORS["border"], graph_rect, 1)
        surf.blit(self.viewer.text_cache.render("Score Lead", 13, COLORS["text_dim"]), (graph_rect.x + 5, graph_rect.y + 2))


class SidePanel:
    def __init__(self, viewer, registry):
        self.viewer, self.registry = viewer, registry

    def _draw_legend(self, surf, x, y):
        y += 20
        surf.blit(self.viewer.text_cache.render("Legend & Controls", 18, COLORS["text"]), (x, y))
        y += 30

        legend_items = {
            "Red Soldier": COLORS["red_soldier"],
            "Red Archer": COLORS["red_archer"],
            "Blue Soldier": COLORS["blue_soldier"],
            "Blue Archer": COLORS["blue_archer"],
        }
        for label, color in legend_items.items():
            pygame.draw.rect(surf, color, (x, y, 15, 15))
            surf.blit(self.viewer.text_cache.render(label, 13, COLORS["text_dim"]), (x + 25, y))
            y += 22

        y += 20

        controls = [
            "WASD / Arrows: Pan Camera",
            "Mouse Wheel: Zoom",
            "Click Agent: Inspect",
            "SPACE: Pause Simulation",
            "+/-: Change Speed",
            "R: Toggle Agent Rays",
            "T: Toggle Threat Vision",
            "B: Toggle HP Bars",
            "N: Toggle Brain Labels",
            "S: Save Selected Brain",
            "F9: Manual Checkpoint Save",   # added to legend
            "F11: Toggle Fullscreen",
        ]
        for line in controls:
            surf.blit(self.viewer.text_cache.render(line, 13, COLORS["text_dim"]), (x, y))
            y += 18
        return y

    def draw(self, surf, state_data):
        srect = self.viewer.layout.side_rect()
        surf.fill(COLORS["side_bg"], srect)
        pygame.draw.rect(surf, COLORS["border"], srect, 2)

        pad, y = 12, srect.y + 12
        x = srect.x + pad
        surf.blit(self.viewer.text_cache.render("Agent Inspector", 18, COLORS["text"]), (x, y))
        y += 30

        slot_id = self.viewer.selected_slot_id
        if slot_id is None:
            surf.blit(self.viewer.text_cache.render("Click an agent to inspect.", 13, COLORS["text_dim"]), (x, y))
            y += 30
        elif slot_id not in state_data["agent_map"]:
            uid_str = f"ID: {self.viewer.last_selected_uid} (Dead)" if self.viewer.last_selected_uid is not None else f"Slot: {slot_id} (Dead)"
            surf.blit(self.viewer.text_cache.render(uid_str, 13, COLORS["warn"]), (x, y))
            y += 30
        else:
            _ax, _ay, _u, _t, unique_id, _btype = state_data["agent_map"][slot_id]
            surf.blit(self.viewer.text_cache.render(f"ID: {int(unique_id)}", 16, COLORS["green"]), (x, y))
            y += 24

            agent_data = self.registry.agent_data[slot_id]
            pos = (int(agent_data[COL_X].item()), int(agent_data[COL_Y].item()))
            hp, hp_max = float(agent_data[COL_HP].item()), float(agent_data[COL_HP_MAX].item())
            atk, vision = float(agent_data[COL_ATK].item()), float(agent_data[COL_VISION].item())
            hp_ratio = (hp / hp_max) if hp_max > 0 else 0.0

            # Unit + Brain labels 
            unit_val = float(agent_data[COL_UNIT].item())
            unit_name = "Archer" if unit_val == 2.0 else "Soldier"

            brain = self.registry.brains[slot_id]
            brain_name = type(brain).__name__ if brain is not None else "<none>"

            agent_score = self.viewer.agent_scores.get(int(unique_id), 0.0)
            surf.blit(self.viewer.text_cache.render(f"Score: {agent_score:.2f}", 13, COLORS["text_dim"]), (x, y)); y += 18
            surf.blit(self.viewer.text_cache.render(f"Pos: ({pos[0]}, {pos[1]})", 13, COLORS["text_dim"]), (x, y)); y += 18

            surf.blit(self.viewer.text_cache.render(f"Unit: {unit_name}", 13, COLORS["text_dim"]), (x, y)); y += 18
            surf.blit(self.viewer.text_cache.render(f"Brain: {brain_name}", 13, COLORS["text_dim"]), (x, y)); y += 18

            bar_w = srect.width - 2 * pad
            pygame.draw.rect(surf, COLORS["bar_bg"], (x, y, bar_w, 10))
            pygame.draw.rect(surf, COLORS["bar_fg"], (x, y, bar_w * hp_ratio, 10))
            y += 14

            surf.blit(self.viewer.text_cache.render(f"HP: {hp:.2f} / {hp_max:.2f}", 13, COLORS["text_dim"]), (x, y)); y += 20
            surf.blit(self.viewer.text_cache.render(f"Attack: {atk:.2f}", 13, COLORS["text_dim"]), (x, y)); y += 18
            surf.blit(self.viewer.text_cache.render(f"Vision: {vision} cells", 13, COLORS["text_dim"]), (x, y)); y += 22

            if brain:
                surf.blit(self.viewer.text_cache.render(f"Model: {_get_model_summary(brain)}", 13, COLORS["text_dim"]), (x, y)); y += 18
                surf.blit(self.viewer.text_cache.render(f"Params: {_param_count(brain):,}", 13, COLORS["text_dim"]), (x, y)); y += 18

        pygame.draw.line(surf, COLORS["border"], (srect.x, y + 10), (srect.right, y + 10), 2)
        self._draw_legend(surf, x, y)


class Minimap:
    def __init__(self, viewer):
        self.viewer = viewer
        self.grid_w = viewer.grid.shape[2]
        self.grid_h = viewer.grid.shape[1]

    def draw(self, surf, hud_rect, state_data):
        map_w = 200
        map_h = int(map_w * (self.grid_h / self.grid_w))
        map_rect = pygame.Rect(hud_rect.right - map_w - 20, hud_rect.y + (hud_rect.height - map_h) // 2, map_w, map_h)
        surf.fill(COLORS["empty"], map_rect)

        for x, y, _unit, team, _uid, _btype in state_data["agent_map"].values():
            dot_x = map_rect.x + int(float(x) / self.grid_w * map_w)
            dot_y = map_rect.y + int(float(y) / self.grid_h * map_h)
            color = COLORS["red"] if team == 2.0 else COLORS["blue"]
            pygame.draw.rect(surf, color, (dot_x, dot_y, 2, 2))

        if self.viewer.cam.cell_px > 0:
            cam_rect_world = pygame.Rect(
                self.viewer.cam.offset_x,
                self.viewer.cam.offset_y,
                self.viewer.layout.world_rect().width / self.viewer.cam.cell_px,
                self.viewer.layout.world_rect().height / self.viewer.cam.cell_px,
            )
            cam_rect_map = pygame.Rect(
                map_rect.x + (cam_rect_world.x / self.grid_w * map_w),
                map_rect.y + (cam_rect_world.y / self.grid_h * map_h),
                (cam_rect_world.width / self.grid_w * map_w),
                (cam_rect_world.height / self.grid_h * map_h),
            )
            pygame.draw.rect(surf, COLORS["marker"], cam_rect_map, 1)

        pygame.draw.rect(surf, COLORS["border"], map_rect, 1)


# =========================
# Input
# =========================
class InputHandler:
    def __init__(self, viewer): self.viewer = viewer

    def handle(self):
        running, advance_tick = True, False
        keys = pygame.key.get_pressed()
        pan_speed = 10.0 / max(1.0, float(self.viewer.cam.cell_px))

        if keys[pygame.K_a] or keys[pygame.K_LEFT]: self.viewer.cam.pan(-pan_speed, 0)
        if keys[pygame.K_d] or keys[pygame.K_RIGHT]: self.viewer.cam.pan(pan_speed, 0)
        if keys[pygame.K_w] or keys[pygame.K_UP]: self.viewer.cam.pan(0, -pan_speed)
        if keys[pygame.K_s] or keys[pygame.K_DOWN]: self.viewer.cam.pan(0, pan_speed)

        wrect = self.viewer.layout.world_rect()

        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False

            elif ev.type == pygame.VIDEORESIZE:
                if not self.viewer.fullscreen:
                    self.viewer.Wpix, self.viewer.Hpix = max(800, ev.w), max(520, ev.h)
                    self.viewer.screen = pygame.display.set_mode((self.viewer.Wpix, self.viewer.Hpix), pygame.RESIZABLE)
                    self.viewer.world_renderer.static_surf = None

            elif ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE:
                    running = False
                elif ev.key == pygame.K_SPACE:
                    self.viewer.paused = not self.viewer.paused
                elif ev.key == pygame.K_PERIOD and self.viewer.paused:
                    advance_tick = True
                elif ev.key == pygame.K_m and self.viewer.selected_slot_id is not None:
                    if self.viewer.selected_slot_id in self.viewer.marked:
                        self.viewer.marked.remove(self.viewer.selected_slot_id)
                    elif len(self.viewer.marked) < 10:
                        self.viewer.marked.append(self.viewer.selected_slot_id)
                elif ev.key == pygame.K_r:
                    self.viewer.show_rays = not self.viewer.show_rays
                elif ev.key == pygame.K_t:
                    self.viewer.threat_vision_mode = not self.viewer.threat_vision_mode
                elif ev.key == pygame.K_b:
                    self.viewer.battle_view_enabled = not self.viewer.battle_view_enabled
                elif ev.key == pygame.K_n:
                    self.viewer.show_brain_types = not self.viewer.show_brain_types
                elif ev.key == pygame.K_s:
                    self.viewer.save_selected_brain()
                elif ev.key == pygame.K_F9:               # added: manual checkpoint save
                    self.viewer.save_requested = True
                elif ev.key == pygame.K_F11:
                    self.viewer.fullscreen = not self.viewer.fullscreen
                    if self.viewer.fullscreen:
                        self.viewer.screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
                    else:
                        self.viewer.screen = pygame.display.set_mode((self.viewer.Wpix, self.viewer.Hpix), pygame.RESIZABLE)
                    self.viewer.world_renderer.static_surf = None
                elif ev.key in (pygame.K_EQUALS, pygame.K_PLUS):
                    self.viewer.speed_multiplier = min(16, self.viewer.speed_multiplier * 2)
                elif ev.key == pygame.K_MINUS:
                    self.viewer.speed_multiplier = max(0.25, self.viewer.speed_multiplier / 2)

            elif ev.type == pygame.MOUSEBUTTONDOWN:
                if ev.button == 1 and wrect.collidepoint(ev.pos):
                    gx, gy = self.viewer.cam.screen_to_world(ev.pos[0] - wrect.x, ev.pos[1] - wrect.y)
                    slot_id = self.viewer.fast_grid_pick_slot(gx, gy)
                    self.viewer.selected_slot_id = slot_id if slot_id is not None and slot_id >= 0 else None
                    if self.viewer.selected_slot_id is not None:
                        self.viewer.last_selected_uid = int(self.viewer.registry.agent_data[self.viewer.selected_slot_id, COL_AGENT_ID].item())
                elif ev.button == 4:
                    self.viewer.cam.zoom_at(1.12)
                    self.viewer.world_renderer.static_surf = None
                elif ev.button == 5:
                    self.viewer.cam.zoom_at(1 / 1.12)
                    self.viewer.world_renderer.static_surf = None

        return running, advance_tick


# =========================
# Animation (kept)
# =========================
class AnimationManager:
    def __init__(self): self.animations = []
    def add(self, anim_type, pos): self.animations.append([anim_type, pos, 10])
    def update(self): self.animations = [[t, p, l - 1] for t, p, l in self.animations if l > 0]
    def draw(self, surf, wrect, cam):
        c = cam.cell_px
        for anim_type, pos, lifetime in self.animations:
            cx, cy = cam.world_to_screen(pos[0], pos[1])
            alpha = int(255 * (lifetime / 10))
            if anim_type == "damage":
                pygame.draw.circle(surf, (*COLORS["red"], alpha),
                                   (wrect.x + cx + c // 2, wrect.y + cy + c // 2),
                                   c // 2, 2)


# =========================
# Viewer (Optimized)
# =========================
class Viewer:
    """
    Main speed fixes:
    - Do NOT do grid.detach().cpu() every frame (GPU sync). We refresh state every N frames.
    - Do NOT call .item() for every field for every alive agent each frame.
      We bulk-copy agent columns to CPU when we refresh state.
    - Keep UI smooth at TARGET_FPS for screen recording.
    """

    def __init__(self, grid: torch.Tensor, cell_size: Optional[int] = None, show_grid: bool = True):
        _center_window()
        pygame.init()
        pygame.display.set_caption("Codex Bellum - Transformer")

        self.grid, self.margin, self.show_grid = grid, 8, show_grid
        self.cam = Camera(int(cell_size or config.CELL_SIZE), grid.shape[2], grid.shape[1])

        H, W = grid.shape[1], grid.shape[2]
        side_min_w, hud_h = 280, 126

        # Prevent too-large windows
        try:
            display_info = pygame.display.Info()
            max_w = display_info.current_w - 80
            max_h = display_info.current_h - 120
        except pygame.error:
            max_w, max_h = 1280, 720

        world_px_w, world_px_h = W * self.cam.cell_px, H * self.cam.cell_px
        init_w = min(max_w, max(1280, world_px_w + side_min_w + 3 * self.margin))
        init_h = min(max_h, max(720, world_px_h + hud_h + 2 * self.margin))

        self.Wpix, self.Hpix = int(init_w), int(init_h)
        self.screen = pygame.display.set_mode((self.Wpix, self.Hpix), pygame.RESIZABLE)

        self.text_cache = TextCache()
        self.clock = pygame.time.Clock()

        # Selection
        self.selected_slot_id: Optional[int] = None
        self.last_selected_uid: Optional[int] = None
        self.marked: List[int] = []

        # Toggles
        self.show_rays = False
        self.paused = False
        self.threat_vision_mode = False
        self.battle_view_enabled = False
        self.show_brain_types = False
        self.fullscreen = False
        self.speed_multiplier = 1.0

        # Scores
        self.agent_scores: Dict[int, float] = collections.defaultdict(float)

        # ---- Checkpoint (manual save) ----
        self.save_requested: bool = False          # set True by F9 hotkey, consumed in run()
        self._ckpt_last_status: str = ""           # optional status message

        # ---- Performance knobs ----
        # Refresh expensive CPU state every N frames.
        # 2 = good default for recording @ 30 FPS (state updates 15 Hz).
        self.STATE_REFRESH_EVERY_FRAMES = int(getattr(config, "VIEWER_STATE_REFRESH_EVERY", 2))
        # If you want selection clicks to feel more “exact”, set this to 1.
        self.PICK_REFRESH_EVERY_FRAMES = int(getattr(config, "VIEWER_PICK_REFRESH_EVERY", 2))

        # Cached CPU state
        self._cached_state_data = None
        self._cached_id_np = None   # for click picking
        self._cached_occ_np = None
        self._cached_alive_indices: List[int] = []
        self._cached_agent_map: Dict[int, tuple] = {}

        self._last_state_refresh_frame = -10
        self._last_pick_refresh_frame = -10

    def save_selected_brain(self):
        if self.selected_slot_id is None or not hasattr(self, "registry"):
            return
        brain = self.registry.brains[self.selected_slot_id]
        if brain:
            tick = self.stats.tick if hasattr(self, "stats") else 0
            uid = self.last_selected_uid if self.last_selected_uid is not None else self.selected_slot_id
            filename = f"brain_agent_{uid}_t_{tick}.pth"
            try:
                torch.save(brain.state_dict(), filename)
                print(f"[Viewer] Saved brain for agent {uid} to '{filename}'")
            except Exception as e:
                print(f"[Viewer] Error saving brain: {e}")

    # --------- Fast click picking (no GPU sync) ----------
    def fast_grid_pick_slot(self, gx: int, gy: int) -> Optional[int]:
        # Use cached id grid if we have it
        if self._cached_id_np is None:
            # fallback: GPU -> CPU sync (avoid if possible)
            try:
                v = int(self.grid[2, gy, gx].item())
                return v if v >= 0 else None
            except Exception:
                return None

        if 0 <= gy < self._cached_id_np.shape[0] and 0 <= gx < self._cached_id_np.shape[1]:
            v = int(self._cached_id_np[gy, gx])
            return v if v >= 0 else None
        return None

    # --------- State refresh (bulk copies) ----------
    def _refresh_state_cpu(self):
        """
        One refresh does:
        - grid channels 0 and 2 -> CPU numpy
        - alive indices -> list[int]
        - bulk-copy agent columns for alive -> build agent_map with brain type included
        """
        with torch.no_grad():
            # Copy only what UI needs (channels 0=occ, 2=id)
            # This still syncs, so we do it only every N frames.
            grid_cpu = self.grid.detach().cpu()
            occ_np = grid_cpu[0].numpy().astype(np.int16, copy=False)
            id_np = grid_cpu[2].numpy().astype(np.int32, copy=False)

            ad = self.registry.agent_data
            alive_mask = ad[:, COL_ALIVE] > 0.5
            alive_idx_t = torch.nonzero(alive_mask).squeeze(1)

            if alive_idx_t.numel() == 0:
                alive_indices: List[int] = []
                agent_map: Dict[int, tuple] = {}
            else:
                alive_indices = alive_idx_t.cpu().tolist()

                # Bulk copy only necessary columns for alive agents
                cols = [COL_X, COL_Y, COL_UNIT, COL_TEAM, COL_AGENT_ID]
                alive_data = ad.index_select(0, alive_idx_t)[:, cols].detach().cpu().numpy()
                brains = self.registry.brains

                # Build map: slot_id -> (x, y, unit, team, uid, btype)
                agent_map = {}
                for k, slot_id in enumerate(alive_indices):
                    x, y, unit, team, uid = alive_data[k]
                    br = brains[slot_id]
                    if br is None: 
                        btype = "?"
                    else:
                        name = type(br).__name__
                        if "Tron" in name: btype = "T"
                        elif "Mirror" in name: btype = "M"
                        elif "Transformer" in name: btype = "Tr"
                        else: btype = "U"

                    agent_map[slot_id] = (float(x), float(y), float(unit), float(team), float(uid), btype)

            self._cached_occ_np = occ_np
            self._cached_id_np = id_np
            self._cached_alive_indices = alive_indices
            self._cached_agent_map = agent_map

            self._cached_state_data = {
                "occ_np": self._cached_occ_np,
                "id_np": self._cached_id_np,
                "alive_indices": self._cached_alive_indices,
                "agent_map": self._cached_agent_map,
            }

    # --------- PPO score hook (vectorized-ish) ----------
    def _install_score_hook(self, engine, registry):
        if not hasattr(engine, "_ppo") or engine._ppo is None:
            return

        original_record_step = engine._ppo.record_step

        def record_step_with_score_tracking(*args, **kwargs):
            agent_ids = kwargs.get("agent_ids")
            rewards = kwargs.get("rewards")
            if agent_ids is not None and rewards is not None and agent_ids.numel() > 0:
                # slot ids (on GPU) -> uids (CPU)
                with torch.no_grad():
                    slot_ids = agent_ids.detach()
                    uids = registry.agent_data.index_select(0, slot_ids)[:, COL_AGENT_ID].detach().cpu().numpy()
                    r = rewards.detach().cpu().numpy()

                # Update dict (still a loop, but now no .item() per element)
                for uid, rv in zip(uids, r):
                    self.agent_scores[int(uid)] += float(rv)

            return original_record_step(*args, **kwargs)

        engine._ppo.record_step = record_step_with_score_tracking

    def run(self, engine, registry, stats, tick_limit: int = 0, target_fps: Optional[int] = None,
            run_dir: Optional[str] = None):   # added run_dir for checkpointing
        self.registry, self.stats = registry, stats

        # Lazy import checkpoint manager only if needed
        ckpt_mgr = None
        if run_dir is not None:
            from utils.checkpointing import CheckpointManager
            ckpt_mgr = CheckpointManager(Path(run_dir))

        self.layout = LayoutManager(self)
        self.world_renderer = WorldRenderer(self, self.grid, registry)
        self.hud_panel = HudPanel(self, stats)
        self.side_panel = SidePanel(self, registry)
        self.input_handler = InputHandler(self)
        self.anim_manager = AnimationManager()
        self.minimap = Minimap(self)

        self.world_renderer.build_static_cache(engine)
        self._install_score_hook(engine, registry)

        running = True
        frame_count = 0

        # Prime cache once so clicks and first frame are fast
        self._refresh_state_cpu()
        self._last_state_refresh_frame = frame_count
        self._last_pick_refresh_frame = frame_count

        while running:
            # Input first (uses cached id_np for click)
            running, advance_tick = self.input_handler.handle()

            # ------------------------------------------------------------
            # Manual atomic checkpoint save (SAFE POINT: between ticks)
            # ------------------------------------------------------------
            if self.save_requested:
                self.save_requested = False
                if ckpt_mgr is None:
                    self._ckpt_last_status = "Checkpoint save requested but run_dir is not set."
                    print("[checkpoint]", self._ckpt_last_status)
                else:
                    try:
                        # Capture viewer state for later restoration
                        viewer_state = {
                            "paused": bool(self.paused),
                            "speed_mult": float(self.speed_multiplier),  # note: attribute is speed_multiplier
                            "camera": {
                                "offset_x": float(getattr(self.cam, "offset_x", 0.0)),
                                "offset_y": float(getattr(self.cam, "offset_y", 0.0)),
                                "zoom": float(getattr(self.cam, "zoom", 1.0)),
                            },
                            "agent_scores": dict(self.agent_scores),
                        }
                        out_dir = ckpt_mgr.save_atomic(
                            engine=engine,
                            registry=registry,
                            stats=stats,
                            viewer_state=viewer_state,
                            notes="manual_hotkey_F9",
                        )
                        self._ckpt_last_status = f"Checkpoint saved: {out_dir.name}"
                        print("[checkpoint]", self._ckpt_last_status)
                    except Exception as ex:
                        self._ckpt_last_status = f"Checkpoint FAILED: {type(ex).__name__}: {ex}"
                        print("[checkpoint]", self._ckpt_last_status)

            # Decide ticks this frame
            num_ticks_this_frame = 0
            if not self.paused:
                if self.speed_multiplier >= 1:
                    num_ticks_this_frame = int(self.speed_multiplier)
                elif self.speed_multiplier > 0:
                    denom = int(1 / self.speed_multiplier)
                    if denom <= 1 or (frame_count % denom) == 0:
                        num_ticks_this_frame = 1
            elif advance_tick:
                num_ticks_this_frame = 1

            # Run sim ticks
            for _ in range(num_ticks_this_frame):
                engine.run_tick()
                self.hud_panel.update()

            # Refresh CPU state sometimes (this is the BIG speed win)
            if (frame_count - self._last_state_refresh_frame) >= self.STATE_REFRESH_EVERY_FRAMES:
                self._refresh_state_cpu()
                self._last_state_refresh_frame = frame_count

            # Optional: refresh pick cache a bit more often if you want
            # (currently we refresh pick with full state refresh, so this is just here for flexibility)
            if (frame_count - self._last_pick_refresh_frame) >= self.PICK_REFRESH_EVERY_FRAMES:
                # id_np is already updated by _refresh_state_cpu; if you ever decouple, handle here.
                self._last_pick_refresh_frame = frame_count

            state_data = self._cached_state_data
            if state_data is None:
                # safety fallback
                self._refresh_state_cpu()
                state_data = self._cached_state_data

            # Draw
            self.screen.fill(COLORS["bg"])
            self.world_renderer.draw(self.screen, state_data)
            self.hud_panel.draw(self.screen, state_data)
            self.side_panel.draw(self.screen, state_data)

            pygame.display.flip()

            # IMPORTANT:
            # - target_fps controls only rendering smoothness.
            # - sim speed is controlled by your speed_multiplier and your config TARGET_TPS (which you set to 0).
            self.clock.tick(int(target_fps or config.TARGET_FPS))

            frame_count += 1
            if tick_limit > 0 and stats.tick >= tick_limit:
                running = False

        pygame.quit()