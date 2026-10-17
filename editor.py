###################################
#                                 #
#   #    #  #####  ####   #####   #
#   #    #    #    #  #   #####   #
#    #  #     #    ### #  #       #
#     #     #####  ####   #####   #
#                                 #
###################################
# vibecoded slop - dont ask me

"""
Minimal NCA editor / viewer.

    editor = NCAEditor(sim, initial_state)   # sim only needs: sim.step(state) -> state
    editor.run()                              # blocking, call from the main thread

State convention: (C, H, W) tensor on whatever device the sim uses, any C >= 1.
A 2D (H, W) tensor is treated as C = 1.

Display: channels 0, 1, 2 are shown as R, G, B. Channels past 2 are hidden.
With 1 channel only R is used, with 2 channels only R and G.

Brush: the color picker sets channels 0, 1, 2 and the "hidden ch" slider sets every channel
past 2 (both in display units, 0..1 across value_range). Erase writes erase_value into every channel.

Tuner (only if the sim has a train method):
    sim.train(world, region, lr, epochs) -> loss
    world is a copy of the full (C, H, W) state, region is (y0, x0, y1, x1), end-exclusive.
    Drag a square on the grid to select it, then press "Train" (or T), or tick
    "train every sim frame" to train once per frame while the sim runs.

Image: drag a rectangle, type a file path (png, jpg, bmp, tga...), press "Load".
    The picture is stretched to fill the rectangle. Its R, G, B go into channels 0, 1, 2,
    channels past 2 get the brush's "hidden ch" value, and transparent pixels keep the world below.

Controls
    Space          play / pause
    Right arrow    single step (works while paused)
    R              reset to initial state
    Brush / Tuner / Image   tool buttons in the panel
    Brush:  left mouse paint, right mouse erase   (paused or running)
    Tuner:  left drag select a square, right click clear, T train
    Image:  left drag select a rectangle, right click clear
"""
from __future__ import annotations

import os
import time
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn.functional as F
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
        tune_lr: float = 1e-3,
        tune_epochs: int = 20,
        min_selection: int = 8,
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
        self.erase_value = float(erase_value)
        # brush color in display units (0..1 across value_range); paint_value sets the starting gray
        start = min(1.0, max(0.0, (float(paint_value) - self.vmin) / (self.vmax - self.vmin)))
        self.paint_color = [start, start, start]   # channels 0, 1, 2
        self.hidden_brightness = start             # channels 3+
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

        # tuner
        self.can_train = callable(getattr(sim, "train", None))
        self.tool = "brush"
        self.tune_lr = float(tune_lr)
        self.tune_epochs = int(tune_epochs)
        self.train_every_frame = False
        self.min_selection = max(1, min(int(min_selection), self.h, self.w))
        self.selection: Optional[tuple[int, int, int, int]] = None  # (y0, x0, y1, x1), end-exclusive
        self._drag_start: Optional[tuple[int, int]] = None           # (y, x) cell where the drag began
        self._drag_rect: Optional[tuple[int, int, int, int]] = None
        self._train_requested = False
        self._last_loss: Optional[float] = None
        self._tuner_msg = ""

        # image import
        self.image_rect: Optional[tuple[int, int, int, int]] = None  # (y0, x0, y1, x1), end-exclusive
        self.image_path = ""
        self._image_msg = ""

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
                if self.tool == "brush":
                    self._handle_brush()
                else:
                    self._handle_select()

                if self._train_requested and self.selection is not None:
                    self._train()
                self._train_requested = False

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
                        if self.train_every_frame and self.selection is not None:
                            self._train()
                        acc -= interval
                        if acc > interval:  # fell behind: drop the backlog, don't spiral
                            acc = 0.0

                if self._dirty:
                    self._upload()
                    self._dirty = False

                self._update_status()
                self._update_overlay()
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

    def _train(self) -> None:
        # outside inference mode, on a plain copy: the sim may build a graph from it
        loss = self.sim.train(self.state.clone(), self.selection,
                              self.tune_lr, self.tune_epochs)
        self._last_loss = None if loss is None else float(loss)

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
        if dpg.is_key_pressed(dpg.mvKey_T) and self.can_train:
            self._train_requested = True

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

        self.state = self.state.clone()  # never mutate a tensor the sim might still hold
        if left:
            self.state[:, mask] = self._paint_vector()[:, None]
        else:
            self.state[:, mask] = self.erase_value  # same value in every channel
        self._dirty = True

    def _paint_vector(self) -> torch.Tensor:
        """Per-channel paint values: picker for channels 0-2, hidden slider for the rest."""
        fracs = self.paint_color[: min(3, self.channels)] + \
            [self.hidden_brightness] * max(0, self.channels - 3)
        fracs = torch.tensor(fracs, dtype=self.state.dtype, device=self.device)
        return self.vmin + fracs * (self.vmax - self.vmin)

    def _set_paint_color(self, sender, *_) -> None:
        r, g, b = dpg.get_value(sender)[:3]          # color edit values are 0..255
        self.paint_color = [r / 255.0, g / 255.0, b / 255.0]

    def _square(self, start: tuple[int, int], cur: tuple[int, int]) -> tuple[int, int, int, int]:
        """Square from the drag start toward the cursor, side = larger drag extent, kept inside the grid."""
        (sy, sx), (cy, cx) = start, cur
        down, right = cy >= sy, cx >= sx
        side = max(abs(cy - sy), abs(cx - sx)) + 1
        side = min(side, (self.h - sy) if down else (sy + 1), (self.w - sx) if right else (sx + 1))
        y0 = sy if down else sy - side + 1
        x0 = sx if right else sx - side + 1
        return y0, x0, y0 + side, x0 + side

    @staticmethod
    def _rect(start: tuple[int, int], cur: tuple[int, int]) -> tuple[int, int, int, int]:
        """Rectangle spanning the drag start and the cursor cell, both included."""
        (sy, sx), (cy, cx) = start, cur
        return min(sy, cy), min(sx, cx), max(sy, cy) + 1, max(sx, cx) + 1

    def _min_size(self) -> int:
        return self.min_selection if self.tool == "tuner" else 1

    def _handle_select(self) -> None:
        """Drag selection for the tuner (square, min size) and image (any rectangle) tools."""
        shape = self._square if self.tool == "tuner" else self._rect
        target = "selection" if self.tool == "tuner" else "image_rect"
        msg = "_tuner_msg" if self.tool == "tuner" else "_image_msg"
        left = dpg.is_mouse_button_down(dpg.mvMouseButton_Left)

        if dpg.is_mouse_button_down(dpg.mvMouseButton_Right) and self._mouse_cell() is not None:
            setattr(self, target, None)
            setattr(self, msg, "")
            self._drag_start, self._drag_rect = None, None
            return

        if left:
            cell = self._mouse_cell()
            if self._drag_start is None:
                if cell is not None:                      # drags must start on the grid
                    self._drag_start = (int(cell[1]), int(cell[0]))
                    self._drag_rect = shape(self._drag_start, self._drag_start)
            elif cell is not None:                        # off-grid: keep the last shape
                self._drag_rect = shape(self._drag_start, (int(cell[1]), int(cell[0])))
            return

        if self._drag_start is not None:                  # button released: commit or reject
            y0, x0, y1, x1 = self._drag_rect
            m = self._min_size()
            if min(y1 - y0, x1 - x0) >= m:
                setattr(self, target, self._drag_rect)
                setattr(self, msg, "")
            else:
                setattr(self, msg, f"too small, min {m}x{m}")
            self._drag_start, self._drag_rect = None, None

    @torch.inference_mode()
    def _load_image(self) -> None:
        if self.image_rect is None:
            self._image_msg = "select a rectangle first"
            return
        path = os.path.expanduser(self.image_path.strip().strip('"').strip("'"))
        if not os.path.isfile(path):
            self._image_msg = "file not found"
            return
        loaded = dpg.load_image(path)
        if loaded is None:
            self._image_msg = "could not read that file as an image"
            return
        w, h, _, data = loaded
        # frombuffer, not asarray: DearPyGui's buffer reports a broken shape on Windows
        # (a 32-bit length read as 64-bit), which numpy turns into "negative dimensions"
        img = np.frombuffer(data, dtype=np.float32, count=w * h * 4)
        img = torch.from_numpy(img.reshape(h, w, 4).copy())
        img = img.permute(2, 0, 1)[None]                                   # (1, 4, h, w), 0..1

        y0, x0, y1, x1 = self.image_rect
        img = F.interpolate(img, size=(y1 - y0, x1 - x0), mode="bilinear",
                            align_corners=False, antialias=True)[0].clamp(0, 1)
        img = img.to(device=self.device, dtype=self.state.dtype)
        rgb, alpha = img[:3], img[3:4]

        # channels 0-2 from the picture, the rest from the brush's hidden value
        fracs = torch.cat([rgb[: min(3, self.channels)],
                           torch.full((max(0, self.channels - 3), *rgb.shape[1:]), self.hidden_brightness,
                                      device=self.device, dtype=self.state.dtype)])
        values = self.vmin + fracs * (self.vmax - self.vmin)

        self.state = self.state.clone()  # never mutate a tensor the sim might still hold
        region = self.state[:, y0:y1, x0:x1]
        self.state[:, y0:y1, x0:x1] = alpha * values + (1 - alpha) * region  # transparent keeps the world
        self._dirty = True
        self._image_msg = f"loaded {w}x{h} -> {x1 - x0}x{y1 - y0}"

    def _set_tool(self, tool: str) -> None:
        if tool == "tuner" and not self.can_train:
            return
        self.tool = tool
        self._last_cell, self._drag_start, self._drag_rect = None, None, None
        for name in ("brush", "tuner", "image"):
            dpg.configure_item(f"{name}_group", show=(tool == name))

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

                    dpg.add_spacer(height=6)
                    dpg.add_separator()
                    with dpg.group(horizontal=True):
                        dpg.add_button(label="Brush", width=65,
                                       callback=lambda: self._set_tool("brush"))
                        dpg.add_button(label="Tuner", width=65, tag="tuner_btn",
                                       callback=lambda: self._set_tool("tuner"))
                        dpg.add_button(label="Image", width=65,
                                       callback=lambda: self._set_tool("image"))
                    if not self.can_train:
                        dpg.disable_item("tuner_btn")
                        dpg.add_text("(sim has no train method)", color=(150, 150, 150))

                    with dpg.group(tag="brush_group"):
                        dpg.add_slider_int(label="brush", default_value=self.brush_radius,
                                           min_value=0, max_value=32, width=150,
                                           callback=lambda s, v: setattr(self, "brush_radius", v))
                        dpg.add_color_edit(label="color", tag="paint_color", width=150,
                                           default_value=[round(c * 255) for c in self.paint_color] + [255],
                                           no_alpha=True, callback=self._set_paint_color)
                        if self.channels > 3:
                            dpg.add_slider_float(label="hidden ch", width=150,
                                                 default_value=self.hidden_brightness,
                                                 min_value=0.0, max_value=1.0,
                                                 callback=lambda s, v: setattr(self, "hidden_brightness", v))

                    with dpg.group(tag="tuner_group", show=False):
                        dpg.add_input_float(label="lr", default_value=self.tune_lr, width=150,
                                            format="%.5f", step=1e-4,
                                            callback=lambda s, v: setattr(self, "tune_lr", max(0.0, v)))
                        dpg.add_input_int(label="epochs", default_value=self.tune_epochs, width=150,
                                          callback=lambda s, v: setattr(self, "tune_epochs", max(1, v)))
                        dpg.add_button(label="Train (T)", width=150,
                                       callback=lambda: setattr(self, "_train_requested", True))
                        dpg.add_checkbox(label="train every sim frame",
                                         callback=lambda s, v: setattr(self, "train_every_frame", v))
                        dpg.add_text("", tag="tuner_status")

                    with dpg.group(tag="image_group", show=False):
                        dpg.add_input_text(hint="path to image file", width=self.PANEL_W - 15,
                                           callback=lambda s, v: setattr(self, "image_path", v))
                        dpg.add_button(label="Load into selection", width=150,
                                       callback=self._load_image)
                        dpg.add_text("", tag="image_status", wrap=self.PANEL_W - 15)

                    dpg.add_spacer(height=6)
                    dpg.add_separator()
                    dpg.add_text("", tag="status")
                    dpg.add_spacer(height=6)
                    dpg.add_text("Space  play/pause\n->     step\nR      reset\n"
                                 "Brush: LMB paint, RMB erase\n"
                                 "Tuner: drag a square,\n       RMB clear, T train\n"
                                 "Image: drag a rectangle,\n       RMB clear",
                                 color=(150, 150, 150))

                with dpg.drawlist(width=self.w * self.scale, height=self.h * self.scale, tag="img"):
                    dpg.draw_image("tex", (0, 0), (self.w * self.scale, self.h * self.scale))
                    dpg.draw_rectangle((0, 0), (0, 0), color=(0, 0, 0, 255), thickness=4,
                                       tag="sel_rect_bg", show=False)   # dark outline for contrast
                    dpg.draw_rectangle((0, 0), (0, 0), color=(255, 255, 0, 255), thickness=2,
                                       tag="sel_rect", show=False)

    def _update_status(self) -> None:
        state = "paused" if self.paused else "running"
        dpg.set_value("status", f"{state}\nstep {self._step_count}\n"
                                f"{self.w}x{self.h}, {self.channels} ch")
        if self.tool in ("tuner", "image"):
            committed = self.selection if self.tool == "tuner" else self.image_rect
            if self._drag_rect is not None:
                y0, x0, y1, x1 = self._drag_rect
                sel = f"selecting {x1 - x0}x{y1 - y0}"
            elif committed is not None:
                y0, x0, y1, x1 = committed
                sel = f"selection {x1 - x0}x{y1 - y0} at ({x0}, {y0})"
            else:
                sel = "no selection"
            if self.tool == "tuner":
                extra = "" if self._last_loss is None else f"\nlast loss {self._last_loss:.5f}"
                msg = self._tuner_msg
            else:
                extra, msg = "", self._image_msg
            dpg.set_value(f"{self.tool}_status", sel + extra + (f"\n{msg}" if msg else ""))

    def _update_overlay(self) -> None:
        """Yellow while dragging (red if too small), green for the committed selection."""
        rect, color = None, None
        if self.tool in ("tuner", "image"):
            committed = self.selection if self.tool == "tuner" else self.image_rect
            if self._drag_rect is not None:
                rect = self._drag_rect
                big_enough = min(rect[2] - rect[0], rect[3] - rect[1]) >= self._min_size()
                color = (255, 255, 0, 255) if big_enough else (255, 60, 60, 255)
            elif committed is not None:
                rect, color = committed, (60, 255, 60, 255)
        if rect is None:
            dpg.configure_item("sel_rect", show=False)
            dpg.configure_item("sel_rect_bg", show=False)
            return
        y0, x0, y1, x1 = rect
        s = self.scale
        corners = dict(pmin=(x0 * s, y0 * s), pmax=(x1 * s - 1, y1 * s - 1))
        dpg.configure_item("sel_rect_bg", show=True, **corners)
        dpg.configure_item("sel_rect", show=True, color=color, **corners)


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