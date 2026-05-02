from __future__ import annotations

import argparse
import base64
import io
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Dict, List, Optional, Tuple
import urllib.parse

import numpy as np
from PIL import Image
from scipy.ndimage import map_coordinates

from straighten import (
    StraightenConfig,
    fit_and_resample_centerline,
    load_and_binarize,
    paste_on_white_canvas,
    straighten_along_normals,
)

N_POINTS = 5
MODE_NAME = "manual_5point_web"

POINT_NAMES = ['upper_left', 'upper_right', 'centromere', 'lower_left', 'lower_right']
POINT_LABELS = ['1 上左肩点 / Upper-left shoulder', '2 上右肩点 / Upper-right shoulder', '3 着丝粒 / Centromere', '4 下左肩点 / Lower-left shoulder', '5 下右肩点 / Lower-right shoulder']


def norm_rel(p: str) -> str:
    return os.path.normpath(p).replace("\\", "/")


def ensure_dir(p: str) -> None:
    if p:
        os.makedirs(p, exist_ok=True)


def load_json(path: str) -> Dict:
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_json(path: str, data: Dict) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def list_images(input_root: str, splits: Optional[List[str]], limit: Optional[int]) -> List[str]:
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    roots = [os.path.join(input_root, s) for s in splits] if splits else [input_root]
    out: List[str] = []
    for root in roots:
        if not os.path.isdir(root):
            print(f"[warn] missing split/root: {root}")
            continue
        for dirpath, _dirnames, filenames in os.walk(root):
            for fn in sorted(filenames):
                if os.path.splitext(fn)[1].lower() in exts:
                    out.append(os.path.join(dirpath, fn))
    out = sorted(out)
    if limit is not None and limit > 0:
        out = out[:limit]
    return out


def _unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v)
    if n < 1e-8:
        return np.zeros_like(v)
    return v / n


def _sample_mask_value(mask: np.ndarray, q_yx: np.ndarray) -> float:
    y, x = float(q_yx[0]), float(q_yx[1])
    return float(map_coordinates(mask.astype(np.float32), [[y], [x]], order=1, mode="constant", cval=0.0)[0])


def _march_to_mask_boundary(
    start_yx: np.ndarray,
    direction_yx: np.ndarray,
    mask: np.ndarray,
    step: float = 0.5,
    max_dist: Optional[float] = None,
) -> np.ndarray:
    """
    Start from an interior shoulder midpoint and move along direction_yx until
    just before exiting the chromosome mask. Return the last in-mask point.
    This prevents cutting off the cap above/below the shoulder line.
    """
    H, W = mask.shape
    p0 = np.asarray(start_yx, dtype=np.float64)
    d = _unit(direction_yx)
    if np.linalg.norm(d) < 1e-8:
        return p0

    if max_dist is None:
        max_dist = float(np.hypot(H, W) + 10.0)

    if _sample_mask_value(mask, p0) < 0.5:
        # If the user clicked slightly outside, do not make things worse.
        return p0

    last_inside = p0.copy()
    first_outside = None
    n_steps = int(np.ceil(max_dist / step))
    for i in range(1, n_steps + 1):
        q = p0 + d * (i * step)
        if q[0] < -1 or q[0] > H or q[1] < -1 or q[1] > W:
            first_outside = q
            break
        if _sample_mask_value(mask, q) >= 0.5:
            last_inside = q
        else:
            first_outside = q
            break

    if first_outside is None:
        return last_inside

    # Refine boundary with binary search.
    lo = last_inside.copy()
    hi = first_outside.copy()
    for _ in range(18):
        mid = 0.5 * (lo + hi)
        if _sample_mask_value(mask, mid) >= 0.5:
            lo = mid
        else:
            hi = mid
    return lo


def _choose_normal_from_pair(p_left: np.ndarray, p_right: np.ndarray, toward_vec: np.ndarray) -> np.ndarray:
    """
    Coordinates are y,x.
    For a left-right boundary pair, choose the normal direction that points toward toward_vec.
    """
    chord = np.asarray(p_right, dtype=np.float64) - np.asarray(p_left, dtype=np.float64)
    normal = np.array([-chord[1], chord[0]], dtype=np.float64)
    normal = _unit(normal)
    toward_vec = _unit(toward_vec)
    if np.dot(normal, toward_vec) < 0:
        normal = -normal
    return normal


def _cubic_hermite(p0: np.ndarray, p1: np.ndarray, m0: np.ndarray, m1: np.ndarray, t: np.ndarray) -> np.ndarray:
    t = t.reshape(-1, 1)
    h00 = 2 * t**3 - 3 * t**2 + 1
    h10 = t**3 - 2 * t**2 + t
    h01 = -2 * t**3 + 3 * t**2
    h11 = t**3 - t**2
    return h00 * p0 + h10 * m0 + h01 * p1 + h11 * m1



def build_manual_path_from_points(points_yx: np.ndarray, mask: np.ndarray, n_per_seg: int = 70, n_cap: int = 22) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    5-point mode. points_yx order:
      0 upper_left
      1 upper_right
      2 centromere
      3 lower_left
      4 lower_right

    Path:
      top_tip -> top_mid -> centromere -> bottom_mid -> bottom_tip

    top_tip / bottom_tip are automatically extended from the shoulder-midpoints
    to the real mask boundary, so the cap above/below the shoulder line is not cut off.
    """
    pts = np.asarray(points_yx, dtype=np.float64)
    if pts.shape != (5, 2):
        raise ValueError("Need exactly 5 points in y,x order")

    p_ul, p_ur, p_c, p_ll, p_lr = pts

    p_top = 0.5 * (p_ul + p_ur)
    p_bottom = 0.5 * (p_ll + p_lr)

    # Inward/outward tangent hints from the pseudo upper/lower boundary lines.
    t_top_in = _choose_normal_from_pair(p_ul, p_ur, p_c - p_top)
    t_bottom_out = _choose_normal_from_pair(p_ll, p_lr, p_bottom - p_c)

    # Extend from shoulder midpoints to the real mask tips.
    p_top_tip = _march_to_mask_boundary(p_top, -t_top_in, mask)
    p_bottom_tip = _march_to_mask_boundary(p_bottom, t_bottom_out, mask)

    v_tc = _unit(p_c - p_top)
    v_cb = _unit(p_bottom - p_c)
    t_center = _unit(v_tc + v_cb)
    if np.linalg.norm(t_center) < 1e-8:
        t_center = _unit(p_bottom - p_top)
    if np.linalg.norm(t_center) < 1e-8:
        t_center = np.array([1.0, 0.0], dtype=np.float64)

    d_top = float(np.linalg.norm(p_c - p_top))
    d_bottom = float(np.linalg.norm(p_bottom - p_c))
    d_mid = min(d_top, d_bottom)

    # Hermite tangent magnitudes.
    m_top = t_top_in * (0.85 * d_top)
    m_bottom = t_bottom_out * (0.85 * d_bottom)
    m_center_top = t_center * (0.45 * d_mid)
    m_center_bottom = t_center * (0.45 * d_mid)

    t = np.linspace(0.0, 1.0, n_per_seg)
    seg1 = _cubic_hermite(p_top, p_c, m_top, m_center_top, t)
    seg2 = _cubic_hermite(p_c, p_bottom, m_center_bottom, m_bottom, t)

    cap_top = np.linspace(p_top_tip, p_top, n_cap)
    cap_bottom = np.linspace(p_bottom, p_bottom_tip, n_cap)

    path_yx = np.vstack([
        cap_top[:-1],
        seg1[:-1],
        seg2[:-1],
        cap_bottom,
    ])

    aux = {
        "upper_left": p_ul,
        "upper_right": p_ur,
        "centromere": p_c,
        "lower_left": p_ll,
        "lower_right": p_lr,
        "top_mid": p_top,
        "bottom_mid": p_bottom,
        "top_tip": p_top_tip,
        "bottom_tip": p_bottom_tip,
    }
    return path_yx, aux



def image_to_png_base64(arr: np.ndarray, mode: str = "L") -> str:
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    bio = io.BytesIO()
    Image.fromarray(arr, mode=mode).save(bio, format="PNG")
    return base64.b64encode(bio.getvalue()).decode("ascii")


def file_image_base64(path: str) -> str:
    with Image.open(path).convert("L") as img:
        bio = io.BytesIO()
        img.save(bio, format="PNG")
    return base64.b64encode(bio.getvalue()).decode("ascii")


def run_manual_straightening(image_path: str, cfg: StraightenConfig, points_yx: np.ndarray) -> Dict:
    gray, mask = load_and_binarize(image_path, cfg)
    base_path, aux = build_manual_path_from_points(points_yx, mask)
    center_pts, tangents, normals = fit_and_resample_centerline(base_path, cfg)
    straight_raw, sample_ys, sample_xs = straighten_along_normals(gray, center_pts, normals, cfg)
    if cfg.output_canvas_size is not None:
        straight = paste_on_white_canvas(straight_raw, cfg.output_canvas_size, fill=cfg.fill_value)
    else:
        straight = straight_raw
    return {
        "gray": gray,
        "mask": mask,
        "base_path": base_path,
        "aux": aux,
        "centerline_pts": center_pts,
        "tangents": tangents,
        "normals": normals,
        "sample_ys": sample_ys,
        "sample_xs": sample_xs,
        "straightened_raw": straight_raw,
        "straightened": straight,
    }


def save_output_image(arr: np.ndarray, out_path: str) -> None:
    ensure_dir(os.path.dirname(out_path))
    Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode="L").save(out_path)


INDEX_HTML = r"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>Chromosome Interactive Straightening</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 16px; background: #f7f7f7; }
    .topbar { display: flex; gap: 12px; align-items: center; margin-bottom: 12px; flex-wrap: wrap; }
    button { padding: 8px 12px; border: 1px solid #aaa; border-radius: 6px; background: white; cursor: pointer; }
    button.primary { background: #1677ff; color: white; border-color: #1677ff; }
    button.danger { background: #fff1f0; color: #a8071a; }
    .panel { display: flex; gap: 18px; align-items: flex-start; }
    .card { background: white; border-radius: 10px; padding: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); }
    canvas { border: 1px solid #ccc; background: white; cursor: crosshair; }
    #preview { border: 1px solid #ccc; background: white; width: 300px; height: 300px; object-fit: contain; }
    .hint { color: #555; font-size: 14px; }
    .status { font-weight: bold; color: #1677ff; }
    .small { font-size: 13px; color: #666; }
    .pointlist { line-height: 1.7; }
  </style>
</head>
<body>
  <div class="topbar">
    <button onclick="prevImage()">上一张</button>
    <button onclick="nextImage()">下一张</button>
    <button onclick="undoPoint()">撤销 U</button>
    <button onclick="resetPoints()" class="danger">重选 R</button>
    <button onclick="saveCurrent()" class="primary">确认保存 Enter</button>
    <button onclick="skipImage()">跳过 N</button>
    <span class="status" id="status"></span>
  </div>

  <div class="card" style="margin-bottom:12px">
    <div><b id="title"></b></div>
    <div class="hint">__INSTRUCTION__</div>
    <div class="small">快捷键：U 撤销，R 重选，Enter 保存，N 跳过，←/→ 切换图片。</div>
  </div>

  <div class="panel">
    <div class="card">
      <h3>原图 + 标注点</h3>
      <canvas id="canvas" width="300" height="300"></canvas>
      <div class="pointlist" id="pointList"></div>
    </div>
    <div class="card">
      <h3>拉直预览</h3>
      <img id="preview" />
      <div class="small" id="previewInfo"></div>
    </div>
  </div>

<script>
const N_POINTS = __N_POINTS__;
const labels = __LABELS__;

let images = [];
let idx = 0;
let points = [];
let img = new Image();
let canvas = document.getElementById("canvas");
let ctx = canvas.getContext("2d");
let previewImg = document.getElementById("preview");
let originalW = 300;
let originalH = 300;

async function api(path, payload=null) {
  const opts = payload ? {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify(payload)
  } : {};
  const r = await fetch(path.replace(/^\//, ""), opts);
  if (!r.ok) {
    const text = await r.text();
    throw new Error(text || r.statusText);
  }
  return await r.json();
}

function setStatus(s) {
  document.getElementById("status").innerText = s;
}

function draw() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

  if (points.length >= 2) {
    ctx.strokeStyle = "rgba(255,0,0,0.82)";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(points[0].x, points[0].y);
    for (let i=1; i<points.length; i++) ctx.lineTo(points[i].x, points[i].y);
    ctx.stroke();
  }

  // top pseudo boundary
  if (points.length >= 2) {
    ctx.strokeStyle = "rgba(0,200,255,0.95)";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(points[0].x, points[0].y);
    ctx.lineTo(points[1].x, points[1].y);
    ctx.stroke();
  }

  // bottom pseudo boundary
  if (N_POINTS === 5 && points.length >= 5) {
    ctx.strokeStyle = "rgba(0,200,255,0.95)";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(points[3].x, points[3].y);
    ctx.lineTo(points[4].x, points[4].y);
    ctx.stroke();
  }
  if (N_POINTS === 7 && points.length >= 7) {
    ctx.strokeStyle = "rgba(0,200,255,0.95)";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(points[5].x, points[5].y);
    ctx.lineTo(points[6].x, points[6].y);
    ctx.stroke();
  }

  for (let i=0; i<points.length; i++) {
    const p = points[i];
    ctx.beginPath();
    ctx.arc(p.x, p.y, 5, 0, 2*Math.PI);
    ctx.fillStyle = (N_POINTS === 5 && i === 2) || (N_POINTS === 7 && i === 3) ? "yellow" : "red";
    ctx.fill();
    ctx.strokeStyle = "black";
    ctx.lineWidth = 1.5;
    ctx.stroke();

    ctx.fillStyle = "black";
    ctx.font = "bold 13px Arial";
    ctx.fillText(String(i+1), p.x + 8, p.y - 8);
  }

  updatePointList();
}

function updatePointList() {
  let html = "";
  for (let i=0; i<N_POINTS; i++) {
    if (i < points.length) {
      html += `<div>${i+1}. ${labels[i]}: x=${points[i].x.toFixed(1)}, y=${points[i].y.toFixed(1)}</div>`;
    } else {
      html += `<div style="color:#999">${i+1}. ${labels[i]}: 未选择</div>`;
    }
  }
  document.getElementById("pointList").innerHTML = html;
}

function canvasToPoint(clientX, clientY) {
  const rect = canvas.getBoundingClientRect();
  const xCanvas = clientX - rect.left;
  const yCanvas = clientY - rect.top;
  return {x: xCanvas, y: yCanvas};
}

canvas.addEventListener("click", async (ev) => {
  if (points.length >= N_POINTS) return;
  const p = canvasToPoint(ev.clientX, ev.clientY);
  points.push(p);
  draw();
  if (points.length === N_POINTS) await previewCurrent();
  else setStatus(`请选择第 ${points.length + 1} 个点：${labels[points.length]}`);
});

function pointsYX() {
  return points.map(p => [
    p.y * originalH / canvas.height,
    p.x * originalW / canvas.width
  ]);
}

async function loadImage(i) {
  if (i < 0 || i >= images.length) return;
  idx = i;
  points = [];
  previewImg.removeAttribute("src");
  document.getElementById("previewInfo").innerText = "";

  const meta = await api(`api/image?index=${idx}`);
  originalW = meta.width;
  originalH = meta.height;
  document.getElementById("title").innerText = `[${idx+1}/${images.length}] ${meta.rel_path}`;

  img.onload = () => {
    canvas.width = 300;
    canvas.height = 300;
    draw();
  };
  img.src = "data:image/png;base64," + meta.image_base64;

  if (meta.existing_points_yx && meta.existing_points_yx.length === N_POINTS) {
    points = meta.existing_points_yx.map(p => ({
      x: p[1] * canvas.width / originalW,
      y: p[0] * canvas.height / originalH,
    }));
    setTimeout(async () => {
      draw();
      await previewCurrent();
    }, 100);
  }
  setStatus(`请选择第 1 个点：${labels[0]}`);
}

async function previewCurrent() {
  if (points.length !== N_POINTS) return;
  setStatus("正在生成预览...");
  try {
    const res = await api("api/preview", {
      index: idx,
      points_yx: pointsYX()
    });
    previewImg.src = "data:image/png;base64," + res.preview_base64;
    document.getElementById("previewInfo").innerText = `raw shape: ${res.raw_shape}, final shape: ${res.final_shape}`;
    setStatus("预览完成：满意请按 Enter 保存");
  } catch (e) {
    setStatus("预览失败：" + e.message);
  }
}

function undoPoint() {
  if (points.length > 0) {
    points.pop();
    previewImg.removeAttribute("src");
    document.getElementById("previewInfo").innerText = "";
    draw();
    setStatus(`请选择第 ${points.length + 1} 个点：${labels[points.length] || ""}`);
  }
}

function resetPoints() {
  points = [];
  previewImg.removeAttribute("src");
  document.getElementById("previewInfo").innerText = "";
  draw();
  setStatus("已重置，请重新选择第 1 个点");
}

async function saveCurrent() {
  if (points.length !== N_POINTS) {
    setStatus(`需要先选择 ${N_POINTS} 个点`);
    return;
  }
  setStatus("正在保存...");
  try {
    const res = await api("api/save", {
      index: idx,
      points_yx: pointsYX()
    });
    setStatus("已保存：" + res.output_rel_path);
    await new Promise(r => setTimeout(r, 400));
    if (idx < images.length - 1) await loadImage(idx + 1);
  } catch (e) {
    setStatus("保存失败：" + e.message);
  }
}

async function skipImage() {
  setStatus("已跳过");
  if (idx < images.length - 1) await loadImage(idx + 1);
}

async function nextImage() {
  if (idx < images.length - 1) await loadImage(idx + 1);
}

async function prevImage() {
  if (idx > 0) await loadImage(idx - 1);
}

document.addEventListener("keydown", async (ev) => {
  if (ev.key === "u" || ev.key === "U") undoPoint();
  else if (ev.key === "r" || ev.key === "R") resetPoints();
  else if (ev.key === "Enter") await saveCurrent();
  else if (ev.key === "n" || ev.key === "N") await skipImage();
  else if (ev.key === "ArrowRight") await nextImage();
  else if (ev.key === "ArrowLeft") await prevImage();
});

async function init() {
  try {
    const res = await api("api/list");
    images = res.images;
    if (images.length === 0) {
      setStatus("没有找到图片");
      return;
    }
    await loadImage(0);
  } catch (e) {
    setStatus("页面初始化失败：" + e.message);
    console.error(e);
  }
}
init();
</script>
</body>
</html>
""".replace("__N_POINTS__", str(N_POINTS)).replace("__LABELS__", json.dumps(['上左肩点', '上右肩点', '着丝粒', '下左肩点', '下右肩点'], ensure_ascii=False)).replace("__INSTRUCTION__", "点击顺序：1 上左肩点 → 2 上右肩点 → 3 着丝粒 → 4 下左肩点 → 5 下右肩点。点满 5 个后右侧自动预览。")


class AppState:
    def __init__(self, args):
        self.input_root = os.path.abspath(args.input_root)
        self.output_root = os.path.abspath(args.output_root)
        ensure_dir(self.output_root)

        self.annotations_path = os.path.abspath(args.annotations or os.path.join(self.output_root, f"manual_annotations_{N_POINTS}point.json"))
        self.annotations = load_json(self.annotations_path)

        self.images = list_images(self.input_root, args.splits, args.limit)
        self.rel_paths = [norm_rel(os.path.relpath(p, self.input_root)) for p in self.images]
        self.overwrite = bool(args.overwrite)

        self.cfg = StraightenConfig(
            bg_threshold=args.bg_threshold,
            normal_half_width=args.normal_half_width,
            normal_step=args.normal_step,
            spline_smoothing_scale=args.spline_smoothing_scale,
            output_canvas_size=tuple(args.output_canvas_size),
        )

    def image_path(self, index: int) -> str:
        return self.images[index]

    def rel_path(self, index: int) -> str:
        return self.rel_paths[index]

    def out_path(self, index: int) -> str:
        return os.path.join(self.output_root, self.rel_path(index))


def make_handler(state: AppState):
    class Handler(BaseHTTPRequestHandler):
        def _send_json(self, obj: Dict, code: int = 200):
            body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_html(self, html: str):
            body = html.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _read_json(self) -> Dict:
            n = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(n)
            return json.loads(raw.decode("utf-8"))

        def log_message(self, fmt, *args):
            print("[web]", fmt % args)

        def do_GET(self):
            parsed = urllib.parse.urlparse(self.path)
            path = parsed.path
            qs = urllib.parse.parse_qs(parsed.query)

            try:
                if path == "/" or path == "/index.html":
                    self._send_html(INDEX_HTML)
                    return

                if path == "/api/list":
                    self._send_json({"images": state.rel_paths, "count": len(state.rel_paths), "n_points": N_POINTS})
                    return

                if path == "/api/image":
                    idx = int(qs.get("index", ["0"])[0])
                    if not (0 <= idx < len(state.images)):
                        self._send_json({"error": "index out of range"}, 400)
                        return
                    path_img = state.image_path(idx)
                    with Image.open(path_img).convert("L") as im:
                        w, h = im.size
                    rel = state.rel_path(idx)
                    existing = state.annotations.get(rel, {})
                    self._send_json({
                        "index": idx,
                        "rel_path": rel,
                        "width": w,
                        "height": h,
                        "image_base64": file_image_base64(path_img),
                        "existing_points_yx": existing.get("points_yx") if isinstance(existing, dict) else None,
                    })
                    return

                self._send_json({"error": "not found"}, 404)
            except Exception as e:
                self._send_json({"error": f"{type(e).__name__}: {e}"}, 500)

        def do_POST(self):
            parsed = urllib.parse.urlparse(self.path)
            path = parsed.path

            try:
                data = self._read_json()
                idx = int(data["index"])
                points_yx = np.asarray(data["points_yx"], dtype=np.float64)

                if points_yx.shape != (N_POINTS, 2):
                    self._send_json({"error": f"points_yx must be shape ({N_POINTS},2)"}, 400)
                    return
                if not (0 <= idx < len(state.images)):
                    self._send_json({"error": "index out of range"}, 400)
                    return

                image_path = state.image_path(idx)
                result = run_manual_straightening(image_path, state.cfg, points_yx)

                if path == "/api/preview":
                    self._send_json({
                        "preview_base64": image_to_png_base64(result["straightened"], "L"),
                        "raw_shape": list(result["straightened_raw"].shape),
                        "final_shape": list(result["straightened"].shape),
                    })
                    return

                if path == "/api/save":
                    out_path = state.out_path(idx)
                    if os.path.exists(out_path) and not state.overwrite:
                        pass
                    else:
                        save_output_image(result["straightened"], out_path)

                    rel = state.rel_path(idx)
                    annotation = {
                        "mode": MODE_NAME,
                        "points_yx": points_yx.tolist(),
                        "point_names": POINT_NAMES,
                        "top_mid_yx": result["aux"]["top_mid"].tolist(),
                        "bottom_mid_yx": result["aux"]["bottom_mid"].tolist(),
                        "top_tip_yx": result["aux"]["top_tip"].tolist(),
                        "bottom_tip_yx": result["aux"]["bottom_tip"].tolist(),
                        "output_image": norm_rel(out_path),
                    }
                    # Add all auxiliary points for easier later analysis.
                    for k, v in result["aux"].items():
                        annotation[f"aux_{k}_yx"] = np.asarray(v, dtype=float).tolist()

                    state.annotations[rel] = annotation
                    save_json(state.annotations_path, state.annotations)

                    self._send_json({
                        "ok": True,
                        "rel_path": rel,
                        "output_path": out_path,
                        "output_rel_path": norm_rel(os.path.relpath(out_path, state.output_root)),
                        "annotations": state.annotations_path,
                    })
                    return

                self._send_json({"error": "not found"}, 404)

            except Exception as e:
                self._send_json({"error": f"{type(e).__name__}: {e}"}, 500)

    return Handler


def main():
    ap = argparse.ArgumentParser(description="Web-based interactive chromosome straightening with 5 manual points.")
    ap.add_argument("--input-root", required=True)
    ap.add_argument("--output-root", required=True)
    ap.add_argument("--splits", nargs="*", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--annotations", default=None)
    ap.add_argument("--overwrite", action="store_true")

    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7860)

    ap.add_argument("--bg-threshold", type=int, default=250)
    ap.add_argument("--normal-half-width", type=int, default=40)
    ap.add_argument("--normal-step", type=float, default=1.0)
    ap.add_argument("--spline-smoothing-scale", type=float, default=3.0)
    ap.add_argument("--output-canvas-size", type=int, nargs=2, default=[300, 300], metavar=("H", "W"))
    args = ap.parse_args()

    state = AppState(args)

    print(f"[info] mode: {MODE_NAME}")
    print(f"[info] found {len(state.images)} images")
    print("[info] point order:")
    for i, label in enumerate(POINT_LABELS, 1):
        print(f"  {i}. {label}")
    print(f"[info] output root: {state.output_root}")
    print(f"[info] annotations: {state.annotations_path}")

    server = ThreadingHTTPServer((args.host, args.port), make_handler(state))
    print(f"[info] open in browser: http://{args.host}:{args.port}")
    print("[info] In VSCode Remote SSH, use PORTS tab to forward this port, then open the forwarded local URL.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[info] stopping server")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
