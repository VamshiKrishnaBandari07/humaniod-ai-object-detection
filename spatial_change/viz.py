"""Point-cloud exports, a still preview, and an offline HTML viewer."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from spatial_change.changes import ChangeResult, resync_summary


def write_ply(path: str | Path, points: np.ndarray, colors: np.ndarray) -> None:
    points = np.asarray(points, dtype=np.float64)
    colors = np.asarray(colors, dtype=np.float64)
    if colors.size and colors.max() <= 1.0:
        colors = colors * 255.0
    colors = np.clip(colors, 0, 255).astype(np.uint8)
    header = (
        "ply\n"
        "format ascii 1.0\n"
        f"element vertex {len(points)}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    )
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(header)
        for point, color in zip(points, colors):
            handle.write(
                f"{point[0]:.4f} {point[1]:.4f} {point[2]:.4f} {int(color[0])} {int(color[1])} {int(color[2])}\n"
            )


def write_report(path: str | Path, result: ChangeResult, metadata: dict) -> None:
    payload = {
        "summary": result.summary,
        "tau_m": result.tau,
        "changes": [
            {
                "type": change.kind,
                "label": change.label,
                "color": change.color_name,
                "description": change.description,
                "size_m": round(change.size, 3),
                "centroid": [round(float(v), 3) for v in change.centroid],
                "centroid_to": None if change.centroid_to is None else [round(float(v), 3) for v in change.centroid_to],
            }
            for change in result.changes
        ],
        "reconstruction": metadata,
    }
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = [
        "# Scene changes",
        "",
        result.summary,
        "",
        "## Detected changes",
        "",
    ]
    if not result.changes:
        lines.append("None.")
    for index, change in enumerate(result.changes, start=1):
        lines.append(f"{index}. **{change.kind}** — {change.description}")
    lines.extend([
        "",
        "## Reconstruction",
        "",
    ])
    for key, value in metadata.items():
        lines.append(f"- {key}: {value}")
    lines.append("")
    Path(path).with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")


def identify_on_frames(result: ChangeResult, before_bgr: np.ndarray, after_bgr: np.ndarray) -> None:
    """Keep a change only when the same colour is a small object in the pictures, then name that place."""
    image_area = float(before_bgr.shape[0] * before_bgr.shape[1])
    kept = []
    for change in result.changes:
        before_box = None if change.kind == "added" else _object_box(before_bgr, change.color_name)
        after_box = None if change.kind == "removed" else _object_box(after_bgr, change.color_name)
        before_area = 0 if before_box is None else before_box[2] * before_box[3]
        after_area = 0 if after_box is None else after_box[2] * after_box[3]
        if max(before_area, after_area) > 0.18 * image_area:
            continue
        if change.kind == "removed" and after_area > max(800, 0.45 * before_area):
            continue
        if change.kind == "added" and before_area > max(800, 0.45 * after_area):
            continue
        sentence = _picture_sentence(change, before_box, after_box, before_bgr.shape)
        if sentence:
            change.description = sentence
        kept.append(change)
    result.changes = kept
    resync_summary(result)


def draw_identification(image_bgr: np.ndarray, result: ChangeResult, visit: str) -> np.ndarray:
    """Box every identified object on one recording."""
    canvas = image_bgr.copy()
    for change in result.changes:
        if visit == "before" and change.kind == "added":
            continue
        if visit == "after" and change.kind == "removed":
            continue
        box = _object_box(image_bgr, change.color_name)
        if box is None:
            continue
        tag = {"moved": "MOVED", "added": "ADDED", "removed": "REMOVED"}[change.kind]
        if change.kind == "moved":
            tag = "FROM" if visit == "before" else "TO"
        _draw_tag(canvas, box, f"{tag}  {change.color_name.upper()}", _bgr(change.rgb))
    return canvas


def write_preview(
    path: str | Path,
    before_bgr: np.ndarray,
    after_bgr: np.ndarray,
    points_a: np.ndarray,
    colors_a: np.ndarray,
    points_b: np.ndarray,
    colors_b: np.ndarray,
    result: ChangeResult,
) -> None:
    """The answer sheet: both recordings with the identified objects boxed, then the sentences."""
    del points_a, colors_a, points_b, colors_b
    canvas = np.full((820, 1280, 3), 16, dtype=np.uint8)
    _label(canvas, "Identified changes", 28, 32, scale=0.7)
    _label(canvas, "Earlier visit", 28, 56, scale=0.45)
    _label(canvas, "Later visit", 652, 56, scale=0.45)
    left = _fit(before_bgr, 600, 340)
    right = _fit(after_bgr, 600, 340)
    top = 68
    canvas[top:top + left.shape[0], 28:28 + left.shape[1]] = left
    canvas[top:top + right.shape[0], 652:652 + right.shape[1]] = right
    y = top + left.shape[0] + 36
    if not result.changes:
        _label(canvas, result.summary, 28, y, scale=0.5)
    for change in result.changes:
        chip = _bgr(change.rgb)
        cv2.rectangle(canvas, (28, y - 14), (46, y + 4), chip, -1)
        _label(canvas, change.kind.upper(), 56, y, scale=0.5)
        for line in _wrap(change.description, 108):
            y += 22
            _label(canvas, line, 56, y, scale=0.48)
            if y > 750:
                break
        y += 28
        if y > 750:
            break
    cv2.imwrite(str(path), canvas)


def write_viewer(path: str | Path, layers: dict[str, tuple[np.ndarray, np.ndarray]], result: ChangeResult) -> None:
    """Self-contained viewer. It uses the canvas element, so it opens from disk with no server."""
    payload = {
        "summary": result.summary,
        "changes": [
            {
                "type": change.kind,
                "label": change.label,
                "description": change.description,
                "centroid": np.round(change.centroid, 3).tolist(),
                "centroid_to": None if change.centroid_to is None else np.round(change.centroid_to, 3).tolist(),
                "rgb": np.round(np.clip(change.rgb, 0, 1) * 255).astype(int).tolist(),
            }
            for change in result.changes
        ],
        "layers": {},
    }
    for name, (points, colors) in layers.items():
        payload["layers"][name] = _pack_layer(points, colors, limit=7000)
    blob = json.dumps(payload).replace("<", "\\u003c")
    html = _VIEWER_HTML.replace("/*DATA*/", blob)
    destination = Path(path)
    destination.write_text(html, encoding="utf-8")
    destination.with_name("scene.json").write_text(json.dumps(payload), encoding="utf-8")


def _pack_layer(points: np.ndarray, colors: np.ndarray, limit: int) -> list[list[float]]:
    points = np.asarray(points, dtype=np.float64)
    colors = np.asarray(colors, dtype=np.float64)
    if len(points) == 0:
        return []
    if colors.max() <= 1.0:
        colors = colors * 255.0
    if len(points) > limit:
        choice = np.linspace(0, len(points) - 1, limit).astype(int)
        points = points[choice]
        colors = colors[choice]
    packed = np.concatenate([np.round(points, 3), np.clip(colors, 0, 255).astype(int)], axis=1)
    return packed.tolist()


def _object_box(image_bgr: np.ndarray, name: str) -> tuple[int, int, int, int] | None:
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    hue = hsv[:, :, 0].astype(np.float64) / 180.0
    saturation = hsv[:, :, 1].astype(np.float64) / 255.0
    value = hsv[:, :, 2].astype(np.float64) / 255.0
    mask = (saturation >= 0.42) & (value >= 0.22) & _hue_is(hue, value, name)
    mask_u8 = mask.astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < 120:
        return None
    x, y, width, height = cv2.boundingRect(contour)
    return int(x), int(y), int(width), int(height)


def _hue_is(hue: np.ndarray, value: np.ndarray, name: str) -> np.ndarray:
    if name == "red":
        return (hue < 0.045) | (hue >= 0.95)
    if name == "orange":
        return (hue >= 0.045) & (hue < 0.09)
    if name == "yellow":
        return (hue >= 0.09) & (hue < 0.18)
    if name == "brown":
        return (hue >= 0.18) & (hue < 0.25) & (value < 0.55)
    if name == "green":
        return (hue >= 0.25) & (hue < 0.48)
    if name == "cyan":
        return (hue >= 0.48) & (hue < 0.58)
    if name == "blue":
        return (hue >= 0.58) & (hue < 0.78)
    if name == "purple":
        return (hue >= 0.78) & (hue < 0.94)
    return np.zeros(hue.shape, dtype=bool)


def _picture_sentence(change, before_box, after_box, shape) -> str | None:
    width = shape[1]

    def side(box) -> str:
        center = (box[0] + box[2] / 2.0) / width
        if center < 0.40:
            return "left"
        if center > 0.60:
            return "right"
        return "middle"

    def noun(name: str) -> str:
        if name == "middle":
            return "the middle"
        return f"the {name} side"

    def placed(name: str) -> str:
        if name == "middle":
            return "in the middle"
        return f"on the {name} side"

    label = change.label
    if change.kind == "moved" and before_box is not None and after_box is not None:
        sentence = (
            f"A {label} moved from {noun(side(before_box))} of the first recording "
            f"to {noun(side(after_box))} of the second."
        )
        if change.centroid_to is not None:
            shift = float(np.linalg.norm(np.asarray(change.centroid_to) - np.asarray(change.centroid)))
            sentence += f" In the shared map that shift is about {shift:.2f} m."
        return sentence
    if change.kind == "removed" and before_box is not None:
        return (
            f"A {label} {placed(side(before_box))} of the first recording was removed. "
            "It is not in the second recording."
        )
    if change.kind == "added" and after_box is not None:
        return (
            f"A {label} appeared {placed(side(after_box))} of the second recording. "
            "It was not in the first."
        )
    return None


def _draw_tag(image: np.ndarray, box: tuple[int, int, int, int], text: str, color: tuple[int, int, int]) -> None:
    x, y, width, height = box
    x0 = max(x - 4, 1)
    y0 = max(y - 4, 1)
    x1 = min(x + width + 4, image.shape[1] - 2)
    y1 = min(y + height + 4, image.shape[0] - 2)
    cv2.rectangle(image, (x0, y0), (x1, y1), color, 2)
    scale = 0.45
    thickness = 1
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    label_top = y0 - th - 8
    if label_top < 2:
        label_top = min(y1 + 2, image.shape[0] - th - 8)
    label_bottom = label_top + th + 6
    label_right = min(x0 + tw + 8, image.shape[1] - 1)
    cv2.rectangle(image, (x0, label_top), (label_right, label_bottom), color, -1)
    cv2.putText(image, text, (x0 + 4, label_bottom - 4), cv2.FONT_HERSHEY_SIMPLEX, scale, (16, 18, 14), thickness, cv2.LINE_AA)


def _bgr(rgb: np.ndarray) -> tuple[int, int, int]:
    rgb = np.asarray(rgb, dtype=np.float64)
    if rgb.max() <= 1.0:
        rgb = rgb * 255.0
    rgb = np.clip(rgb, 0, 255).astype(int)
    return int(rgb[2]), int(rgb[1]), int(rgb[0])


def _legend(result: ChangeResult) -> str:
    if not result.changes:
        return "Top-down"
    bits = [f"{change.color_name} {change.kind}" for change in result.changes]
    return "Top-down  (" + ", ".join(bits) + ")"


def _fit(image: np.ndarray, width: int, height: int) -> np.ndarray:
    scale = min(width / image.shape[1], height / image.shape[0])
    resized = cv2.resize(image, (max(int(image.shape[1] * scale), 1), max(int(image.shape[0] * scale), 1)))
    canvas = np.full((height, width, 3), 18, dtype=np.uint8)
    canvas[: resized.shape[0], : resized.shape[1]] = resized
    return canvas


def _label(image: np.ndarray, text: str, x: int, y: int, scale: float = 0.55) -> None:
    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (235, 235, 235), 1, cv2.LINE_AA)


def _wrap(text: str, width: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        trial = word if not current else current + " " + word
        if len(trial) > width:
            if current:
                lines.append(current)
            current = word
        else:
            current = trial
    if current:
        lines.append(current)
    return lines


def _topdown(
    points_a: np.ndarray,
    colors_a: np.ndarray,
    points_b: np.ndarray,
    colors_b: np.ndarray,
    result: ChangeResult,
    width: int,
    height: int,
) -> np.ndarray:
    canvas = np.full((height, width, 3), 32, dtype=np.uint8)
    stack = []
    if len(points_a):
        stack.append(points_a)
    if len(points_b):
        stack.append(points_b)
    if not stack:
        return canvas
    all_points = np.concatenate(stack)
    lo = np.quantile(all_points[:, :2], 0.02, axis=0)
    hi = np.quantile(all_points[:, :2], 0.98, axis=0)
    span = np.maximum(hi - lo, 1e-3)
    pad = 16

    def paint(points: np.ndarray, colors: np.ndarray, radius: int) -> None:
        if len(points) == 0:
            return
        if len(points) > 6000:
            points = points[:: max(len(points) // 6000, 1)]
            colors = colors[:: max(len(colors) // 6000, 1)]
        cols = colors
        if cols.max() <= 1.0:
            cols = cols * 255.0
        xy = (points[:, :2] - lo) / span
        xs = np.clip((xy[:, 0] * (width - 2 * pad) + pad).astype(int), 0, width - 1)
        ys = np.clip(((1.0 - xy[:, 1]) * (height - 2 * pad) + pad).astype(int), 0, height - 1)
        for x, y, color in zip(xs, ys, cols.astype(np.uint8)):
            cv2.circle(canvas, (int(x), int(y)), radius, (int(color[2]), int(color[1]), int(color[0])), -1)

    gray_a = np.full_like(colors_a, 0.45) if len(colors_a) else colors_a
    gray_b = np.full_like(colors_b, 0.62) if len(colors_b) else colors_b
    paint(points_a, gray_a, 1)
    paint(points_b, gray_b, 1)
    for change in result.changes:
        colour = np.asarray(change.rgb, dtype=np.float64).reshape(1, 3)
        paint(change.points, np.repeat(colour, len(change.points), axis=0), 3)
        if change.points_to is not None:
            paint(change.points_to, np.repeat(colour, len(change.points_to), axis=0), 3)
    return canvas


_VIEWER_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Spatial change viewer</title>
<style>
  :root { color-scheme: dark; }
  body { margin: 0; font-family: "Segoe UI", sans-serif; background: #121418; color: #e8e6e3; }
  header { padding: 16px 22px 8px; }
  h1 { margin: 0 0 6px; font-size: 20px; font-weight: 600; }
  p.summary { margin: 0; max-width: 980px; line-height: 1.45; color: #c8c4bc; }
  main { display: flex; gap: 18px; padding: 12px 22px 28px; align-items: flex-start; }
  .views { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; flex: 1; }
  figure { margin: 0; background: #1c2028; border-radius: 10px; padding: 8px; }
  figcaption { font-size: 12px; letter-spacing: 0.04em; text-transform: uppercase; color: #9a968e; margin-bottom: 6px; }
  canvas { width: 100%; height: auto; background: #101218; border-radius: 6px; }
  aside { width: 320px; }
  label { display: block; margin: 4px 0; font-size: 14px; }
  .change { background: #1c2028; border-radius: 8px; padding: 10px 12px; margin-top: 8px; }
  .change b { text-transform: uppercase; font-size: 11px; letter-spacing: 0.06em; }
  .moved { color: #f0b45a; } .added { color: #67d37a; } .removed { color: #f07167; }
</style>
</head>
<body>
<header>
  <h1>Shared scene, and what changed</h1>
  <p class="summary" id="summary"></p>
</header>
<main>
  <div class="views">
    <figure><figcaption>Top</figcaption><canvas id="top" width="640" height="420"></canvas></figure>
    <figure><figcaption>Front</figcaption><canvas id="front" width="640" height="420"></canvas></figure>
    <figure><figcaption>Side</figcaption><canvas id="side" width="640" height="420"></canvas></figure>
    <figure><figcaption>Legend</figcaption>
      <div style="padding:10px 6px 14px; line-height:1.7">
        <div><span style="color:#9aa0aa">Gray</span> structure seen in either recording</div>
        <div id="legend"></div>
        <div style="color:#9a968e; margin-top:8px">Z is up. Drag is not required — the three views are fixed orthographic maps of the shared frame.</div>
      </div>
    </figure>
  </div>
  <aside>
    <label><input type="checkbox" id="show-before" checked /> Before</label>
    <label><input type="checkbox" id="show-after" checked /> After</label>
    <label><input type="checkbox" id="show-changes" checked /> Changes</label>
    <div id="changes"></div>
  </aside>
</main>
<script>
const SCENE = /*DATA*/;
const summary = document.getElementById("summary");
summary.textContent = SCENE.summary;
const legend = document.getElementById("legend");
legend.innerHTML = SCENE.changes.map(function (change) {
  const rgb = change.rgb || [220, 220, 220];
  return "<div><span style=\\"color:rgb(" + rgb[0] + "," + rgb[1] + "," + rgb[2] + ")\\">" + change.label + "</span> " + change.type + "</div>";
}).join("");
const list = document.getElementById("changes");
for (const change of SCENE.changes) {
  const div = document.createElement("div");
  div.className = "change";
  div.innerHTML = "<b class=\\"" + change.type + "\\">" + change.type + "</b><div>" + change.description + "</div>";
  list.appendChild(div);
}
const toggles = ["show-before", "show-after", "show-changes"];
for (const id of toggles) document.getElementById(id).addEventListener("change", draw);

function bounds() {
  const pts = [];
  for (const layer of Object.values(SCENE.layers)) for (const p of layer) pts.push(p);
  if (!pts.length) return {min:[0,0,0], max:[1,1,1]};
  const min = [Infinity, Infinity, Infinity], max = [-Infinity, -Infinity, -Infinity];
  for (const p of pts) for (let k = 0; k < 3; k++) { min[k] = Math.min(min[k], p[k]); max[k] = Math.max(max[k], p[k]); }
  return {min, max};
}
const BOX = bounds();

function paint(canvas, axisX, axisY) {
  const ctx = canvas.getContext("2d");
  const w = canvas.width, h = canvas.height;
  ctx.clearRect(0, 0, w, h);
  const spanX = Math.max(BOX.max[axisX] - BOX.min[axisX], 1e-3);
  const spanY = Math.max(BOX.max[axisY] - BOX.min[axisY], 1e-3);
  function xy(p) {
    const x = (p[axisX] - BOX.min[axisX]) / spanX;
    const y = (p[axisY] - BOX.min[axisY]) / spanY;
    return [18 + x * (w - 36), h - 18 - y * (h - 36)];
  }
  function dots(layer, color, radius) {
    const pts = SCENE.layers[layer] || [];
    for (const p of pts) {
      const [x, y] = xy(p);
      ctx.fillStyle = color || ("rgb(" + p[3] + "," + p[4] + "," + p[5] + ")");
      ctx.fillRect(x, y, radius, radius);
    }
  }
  if (document.getElementById("show-before").checked) dots("before", "rgba(150,154,162,0.55)", 1.6);
  if (document.getElementById("show-after").checked) dots("after", "rgba(190,194,200,0.45)", 1.6);
  if (document.getElementById("show-changes").checked) {
    dots("removed", null, 3.2);
    dots("added", null, 3.2);
    dots("moved", null, 3.2);
    ctx.strokeStyle = "rgb(240,170,70)";
    ctx.lineWidth = 2;
    for (const change of SCENE.changes) {
      if (change.type !== "moved" || !change.centroid_to) continue;
      const a = xy(change.centroid), b = xy(change.centroid_to);
      ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); ctx.stroke();
    }
  }
}
function draw() {
  paint(document.getElementById("top"), 0, 1);
  paint(document.getElementById("front"), 0, 2);
  paint(document.getElementById("side"), 1, 2);
}
draw();
</script>
</body>
</html>
"""
