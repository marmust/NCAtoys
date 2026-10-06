"""
Minimal NCA editor / viewer.

    editor = NCAEditor(sim, initial_state)   # sim only needs: sim.step(state) -> state
    editor.run()                              # blocking, call from the main thread

State convention: (C, H, W) tensor on whatever device the sim uses, any C >= 1.
A 2D (H, W) tensor is treated as C = 1.

Display: channels 0, 1, 2 are shown as R, G, B. Channels past 2 are hidden.
With 1 channel only R is used, with 2 channels only R and G.

Brush: paints / erases the same value into every channel (so it shows as white / black).

Controls
    Space          play / pause
    Right arrow    single step (works while paused)
    R              reset to initial state
    Left mouse     paint   (paused or running)
    Right mouse    erase
"""
from __future__ import annotations

import time
from typing import Callable, Optional

import numpy as np
import torch
import dearpygui.dearpygui as dpg


class NCAEditor:
    PANEL_W = 230

    def __init__(
        self,
        sim,
        initial_state: torch.Tensor,
        *,
        fps: float = 30.0,
        steps_per_frame: int = 1,
        scale: int = 2,
        value_range: tuple[float, float] = (0.0, 1.0),
        paint_value: float = 1.0,
        erase_value: float = 0.0,
        brush_radius: int = 2,
        start_paused: bool = False,
        to_rgba: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        title: str = "NCA Editor",
    ):
        if initial_state.dim() == 2:
            initial_state = initial_state[None]
        if initial_state.dim() != 3 or initial_state.shape[0] < 1:
            raise ValueError(
                f"expected state of shape (H, W) or (C, H, W), got {tuple(initial_state.shape)}"
            )

        self.sim = sim
        self.initial_state = initial_state.detach().clone()
        self.state = self.initial_state.clone()
        self.channels, self.h, self.w = self.state.shape
        self.device = self.state.device

        self.fps = float(fps)
        self.steps_per_frame = int(steps_per_frame)
        self.scale = int(scale)
        self.vmin, self.vmax = map(float, value_range)
        self.paint_value = float(paint_value)
        self.erase_value = float(erase_value)
        self.brush_radius = int(brush_radius)
        self.paused = bool(start_paused)
        self.title = title
        self._to_rgba = to_rgba or self._default_to_rgba

        # cell-center coordinates for brush masks, kept on the sim's device
        ys = torch.arange(self.h, device=self.device, dtype=torch.float32) + 0.5
        xs = torch.arange(self.w, device=self.device, dtype=torch.float32) + 0.5
        self._yy, self._xx = torch.meshgrid(ys, xs, indexing="ij")

        self._tex_buf = np.zeros(self.h * self.w * 4, dtype=np.float32)
        self._step_count = 0
        self._dirty = True
        self._step_requested = False
        self._reset_requested = False
        self._last_cell: Optional[tuple[float, float]] = None  # for stroke interpolation

    # ------------------------------------------------------------------ public

    def run(self) -> None:
        dpg.create_context()
        dpg.configure_app(manual_callback_management=True)  # callbacks run on this thread
        self._build_ui()
        dpg.create_viewport(
            title=self.title,
            width=self.PANEL_W + self.w * self.scale + 40,
            height=max(self.h * self.scale, 320) + 60,
        )
        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.set_primary_window("main", True)

        last = time.perf_counter()
        acc = 0.0
        try:
            while dpg.is_dearpygui_running():
                dpg.run_callbacks(dpg.get_callback_queue())

                now = time.perf_counter()
                dt, last = now - last, now

                self._handle_keys()
                self._handle_brush()

                if self._reset_requested:
                    self.state = self.initial_state.clone()
                    self._step_count = 0
                    self._reset_requested = False
                    self._dirty = True

                if self._step_requested:
                    self._advance(1)
                    self._step_requested = False

                if self.paused:
                    acc = 0.0
                else:
                    acc += dt
                    interval = 1.0 / self.fps
                    if acc >= interval:
                        self._advance(self.steps_per_frame)
                        acc -= interval
                        if acc > interval:  # fell behind: drop the backlog, don't spiral
                            acc = 0.0

                if self._dirty:
                    self._upload()
                    self._dirty = False

                self._update_status()
                dpg.render_dearpygui_frame()
        finally:
            dpg.destroy_context()

    # -------------------------------------------------------------- simulation

    @torch.inference_mode()
    def _advance(self, n: int) -> None:
        for _ in range(n):
            new = self.sim.step(self.state)
            if new.shape != self.state.shape:
                raise RuntimeError(
                    f"sim.step returned shape {tuple(new.shape)}, expected {tuple(self.state.shape)}"
                )
            self.state = new
        self._step_count += n
        self._dirty = True

    # ----------------------------------------------------------------- display

    def _default_to_rgba(self, state: torch.Tensor) -> torch.Tensor:
        # first up-to-3 channels -> R, G, B; missing colors stay 0
        rgb = ((state[:3] - self.vmin) / (self.vmax - self.vmin)).clamp(0, 1)  # (<=3, H, W)
        rgba = torch.zeros(self.h, self.w, 4, device=state.device, dtype=rgb.dtype)
        rgba[..., : rgb.shape[0]] = rgb.permute(1, 2, 0)
        rgba[..., 3] = 1.0
        return rgba  # (H, W, 4)

    @torch.inference_mode()
    def _upload(self) -> None:
        rgba = self._to_rgba(self.state).float().clamp(0, 1)
        if rgba.shape != (self.h, self.w, 4):
            raise RuntimeError(f"to_rgba must return (H, W, 4), got {tuple(rgba.shape)}")
        np.copyto(self._tex_buf, rgba.cpu().numpy().reshape(-1))
        dpg.set_value("tex", self._tex_buf)

    # ------------------------------------------------------------------- input

    def _handle_keys(self) -> None:
        if dpg.is_key_pressed(dpg.mvKey_Spacebar):
            self._toggle_pause()
        if dpg.is_key_pressed(dpg.mvKey_Right):
            self._step_requested = True
        if dpg.is_key_pressed(dpg.mvKey_R):
            self._reset_requested = True

    def _mouse_cell(self) -> Optional[tuple[float, float]]:
        """Mouse position in (fractional) cell coordinates, or None if off-grid."""
        if not dpg.is_item_hovered("img"):
            return None
        mx, my = dpg.get_mouse_pos(local=False)
        x0, y0 = dpg.get_item_rect_min("img")
        cx, cy = (mx - x0) / self.scale, (my - y0) / self.scale
        if 0 <= cx < self.w and 0 <= cy < self.h:
            return cx, cy
        return None

    @torch.inference_mode()
    def _handle_brush(self) -> None:
        left = dpg.is_mouse_button_down(dpg.mvMouseButton_Left)
        right = dpg.is_mouse_button_down(dpg.mvMouseButton_Right)
        cell = self._mouse_cell() if (left or right) else None
        if cell is None:
            self._last_cell = None
            return

        # interpolate between last and current position so fast drags leave no gaps
        points = [cell]
        if self._last_cell is not None:
            (x0, y0), (x1, y1) = self._last_cell, cell
            dist = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
            spacing = max(0.5, self.brush_radius * 0.5)
            n = int(dist / spacing)
            points = [(x0 + (x1 - x0) * t / (n + 1), y0 + (y1 - y0) * t / (n + 1))
                      for t in range(1, n + 2)]
        self._last_cell = cell

        r2 = (self.brush_radius + 0.5) ** 2
        mask = torch.zeros(self.h, self.w, dtype=torch.bool, device=self.device)
        for cx, cy in points:
            mask |= (self._xx - cx) ** 2 + (self._yy - cy) ** 2 <= r2

        value = self.paint_value if left else self.erase_value
        self.state = self.state.clone()  # never mutate a tensor the sim might still hold
        self.state[:, mask] = value  # same value in every channel
        self._dirty = True

    # ---------------------------------------------------------------------- UI

    def _toggle_pause(self, *_):
        self.paused = not self.paused
        dpg.set_item_label("play_btn", "Play" if self.paused else "Pause")

    def _build_ui(self) -> None:
        with dpg.texture_registry():
            dpg.add_raw_texture(self.w, self.h, self._tex_buf,
                                format=dpg.mvFormat_Float_rgba, tag="tex")

        with dpg.window(tag="main"):
            with dpg.group(horizontal=True):
                with dpg.child_window(width=self.PANEL_W, border=False):
                    with dpg.group(horizontal=True):
                        dpg.add_button(label="Play" if self.paused else "Pause",
                                       tag="play_btn", width=65,
                                       callback=self._toggle_pause)
                        dpg.add_button(label="Step", width=65,
                                       callback=lambda: setattr(self, "_step_requested", True))
                        dpg.add_button(label="Reset", width=65,
                                       callback=lambda: setattr(self, "_reset_requested", True))
                    dpg.add_spacer(height=6)
                    dpg.add_slider_float(label="fps", default_value=self.fps,
                                         min_value=1, max_value=120, width=150,
                                         callback=lambda s, v: setattr(self, "fps", max(1.0, v)))
                    dpg.add_slider_int(label="steps/frame", default_value=self.steps_per_frame,
                                       min_value=1, max_value=64, width=150,
                                       callback=lambda s, v: setattr(self, "steps_per_frame", v))
                    dpg.add_slider_int(label="brush", default_value=self.brush_radius,
                                       min_value=0, max_value=32, width=150,
                                       callback=lambda s, v: setattr(self, "brush_radius", v))
                    dpg.add_input_float(label="paint", default_value=self.paint_value,
                                        width=150, step=0.1,
                                        callback=lambda s, v: setattr(self, "paint_value", v))
                    dpg.add_spacer(height=6)
                    dpg.add_text("", tag="status")
                    dpg.add_spacer(height=6)
                    dpg.add_text("Space  play/pause\n->     step\nR      reset\n"
                                 "LMB    paint\nRMB    erase", color=(150, 150, 150))
                dpg.add_image("tex", tag="img",
                              width=self.w * self.scale, height=self.h * self.scale)

    def _update_status(self) -> None:
        state = "paused" if self.paused else "running"
        dpg.set_value("status", f"{state}\nstep {self._step_count}\n"
                                f"{self.w}x{self.h}, {self.channels} ch")


# --------------------------------------------------------------------- test sim

class GameOfLife:
    """Single-channel test sim, no network. Wraps around the edges."""

    def __init__(self, device):
        k = torch.ones(1, 1, 3, 3, device=device)
        k[0, 0, 1, 1] = 0
        self.kernel = k

    def step(self, state: torch.Tensor) -> torch.Tensor:
        x = (state[None] > 0.5).float()                          # (1, 1, H, W)
        padded = torch.nn.functional.pad(x, (1, 1, 1, 1), mode="circular")
        n = torch.nn.functional.conv2d(padded, self.kernel)
        alive = (n == 3) | ((x == 1) & (n == 2))
        return alive.float()[0]                                  # (1, H, W)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    init = (torch.rand(1, 96, 128, device=device) < 0.25).float()
    NCAEditor(GameOfLife(device), init, fps=20, scale=6).run()