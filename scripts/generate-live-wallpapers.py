#!/usr/bin/env python3
"""
Procedural live-wallpaper generator for Wallpaper-Bank.

Every clip is rendered frame by frame with numpy and piped straight into
ffmpeg. All motion is periodic in the clip length, so the last frame flows
into the first and `mpv --loop` / mpvpaper plays without a visible seam.

    scripts/generate-live-wallpapers.py --list
    scripts/generate-live-wallpapers.py all
    scripts/generate-live-wallpapers.py aurora-lake
    scripts/generate-live-wallpapers.py firefly-forest --size 1440x2560 --seconds 30
    scripts/generate-live-wallpapers.py neon-horizon --preview 0.25     # quick PNG stills

Output lands in wallpapers/Live-Wallpapers/{Horizontal,Vertical}/ unless
--out is given. Requires python3, numpy, pillow and ffmpeg with libx264.
"""
from __future__ import annotations

import argparse
import math
import multiprocessing as mp
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

F32 = np.float32
TAU = 2.0 * math.pi
REPO = Path(__file__).resolve().parents[1]
OUT_ROOT = REPO / "wallpapers" / "Live-Wallpapers"


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------
def hexrgb(h: str) -> np.ndarray:
    h = h.lstrip("#")
    return np.array([int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4)], F32)


def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def mixc(a, b, t):
    """Blend colours a -> b by the scalar field t (any shape)."""
    t = np.asarray(t, F32)[..., None]
    return a + (b - a) * t


def gradient(stops, t):
    """Piecewise colour ramp: stops = [(pos, colour), ...] sorted by pos."""
    t = np.asarray(t, F32)
    out = np.empty(t.shape + (3,), F32)
    out[...] = stops[0][1]
    for (p0, c0), (p1, c1) in zip(stops, stops[1:]):
        w = np.clip((t - p0) / (p1 - p0), 0.0, 1.0)[..., None]
        sel = (t >= p0)[..., None]
        out = np.where(sel, c0 + (c1 - c0) * w, out)
    return out


def resize(a, w, h):
    """Bicubic resize of a float HxW or HxWxC array."""
    a = np.ascontiguousarray(a, dtype=F32)
    if a.ndim == 2:
        return np.array(Image.fromarray(a).resize((w, h), Image.BICUBIC), dtype=F32)
    return np.stack([resize(a[..., c], w, h) for c in range(a.shape[2])], axis=-1)


def _box(a, r, axis):
    if r < 1:
        return a
    a = np.moveaxis(a, axis, 0)
    n = a.shape[0]
    pad = np.concatenate([np.repeat(a[:1], r + 1, 0), a, np.repeat(a[-1:], r, 0)], 0)
    c = np.cumsum(pad, axis=0, dtype=np.float64)
    out = ((c[2 * r + 1:2 * r + 1 + n] - c[:n]) / (2 * r + 1)).astype(F32)
    return np.moveaxis(out, 0, axis)


def blur(a, sigma):
    """Approximate gaussian blur (three box passes). Use on small arrays."""
    if sigma < 0.5:
        return a
    r = max(1, int(round((math.sqrt(4.0 * sigma * sigma + 1.0) - 1.0) / 2.0)))
    for _ in range(3):
        a = _box(_box(a, r, 0), r, 1)
    return a


def tile_noise(h, w, rng, slope=1.2, sx=1.0, sy=1.0, lo=0.0):
    """Seamlessly tileable gaussian noise (FFT filtered), mean 0 / std 1."""
    spec = np.fft.rfft2(rng.standard_normal((h, w)))
    fy = np.fft.fftfreq(h)[:, None] * h * sy
    fx = np.fft.rfftfreq(w)[None, :] * w * sx
    f = np.sqrt(fx * fx + fy * fy)
    f[0, 0] = 1.0
    spec *= f ** (-slope)
    spec[f < max(lo, 1e-9)] = 0.0
    out = np.fft.irfft2(spec, s=(h, w))
    return ((out - out.mean()) / out.std()).astype(F32)


def noise1d(n, rng, slope=1.0, lo=1, hi=None):
    """Periodic 1D band-limited noise, mean 0 / std 1."""
    spec = np.fft.rfft(rng.standard_normal(n))
    f = np.arange(spec.size, dtype=np.float64)
    f[0] = 1.0
    spec *= f ** (-slope)
    spec[:lo] = 0.0
    if hi is not None:
        spec[hi:] = 0.0
    out = np.fft.irfft(spec, n)
    return ((out - out.mean()) / out.std()).astype(F32)


def sample1d(tex, u):
    """Sample a periodic 1D texture; u is in tile units (1.0 = one period)."""
    n = tex.size
    return np.interp(np.asarray(u, np.float64) * n, np.arange(n), tex, period=n).astype(F32)


def shift_tile(tex, dx, dy=0.0):
    """Periodic, sub-pixel (bilinear) translation of a tileable 2D texture."""
    ix, fx = math.floor(dx), dx - math.floor(dx)
    iy, fy = math.floor(dy), dy - math.floor(dy)
    a = np.roll(tex, (iy, ix), (0, 1))
    row = a * (1.0 - fx) + np.roll(a, 1, 1) * fx
    return row * (1.0 - fy) + np.roll(row, 1, 0) * fy


def ss_mask(w, h, ss, draw_fn):
    """Draw polygons on a supersampled 8-bit canvas, return an AA float mask."""
    im = Image.new("L", (w * ss, h * ss), 0)
    draw_fn(ImageDraw.Draw(im), ss)
    if ss > 1:
        im = im.reduce(ss)
    return np.asarray(im, dtype=F32) / 255.0


def pine(x, base, height, width, rng):
    """Outline of a stylised spruce as a list of (x, y) points."""
    tiers = int(rng.integers(8, 14))
    trunk = max(width * 0.05, 1.0)
    left, right = [(x - trunk, base)], [(x + trunk, base)]
    th = height * 0.92 / tiers
    for i in range(tiers):
        frac = i / tiers
        y = base - height * 0.08 - th * i
        for side, pts in ((-1, left), (1, right)):
            half = width * 0.5 * (1.0 - frac) ** 1.05 * (0.8 + 0.4 * rng.random())
            droop = th * (0.15 + 0.25 * rng.random())
            pts.append((x + side * half, y + droop))
            pts.append((x + side * half * 0.32, y - th * 0.55))
    top = (x, base - height)
    return left + [top] + right[::-1]


def ridge_mask(ridge_y, y0, y1):
    """Antialiased coverage of everything below ridge_y (per column), rows y0..y1."""
    rows = np.arange(y0, y1, dtype=F32)[:, None]
    return np.clip(rows + 1.0 - ridge_y[None, :], 0.0, 1.0)


class Stars:
    """Sparse star field with twinkle groups; additive into an HxWx3 frame."""

    def __init__(self, W, H, n, rng, s, y0=0, y1=None, density=None, visible=None,
                 groups=8, twinkle=0.45, bright=1.0):
        y1 = H if y1 is None else y1
        xs = rng.uniform(0, W, n)
        ys = rng.uniform(y0, y1, n)
        if density is not None:
            keep = rng.random(n) < density(xs / W, ys / H)
            xs, ys = xs[keep], ys[keep]
        n = xs.size
        mag = rng.pareto(2.4, n)
        b = np.clip(0.12 + 0.30 * mag, 0.0, 2.2) * bright
        sig = (0.55 + 0.30 * np.clip(mag, 0, 2.5)) * max(s, 0.35)
        pal = np.stack([hexrgb(c) for c in ("#ffffff", "#d6e4ff", "#fff3dc", "#ffd9c9", "#cfd8ff")])
        cols = pal[rng.integers(0, len(pal), n)]
        R = 4
        o = np.arange(-R, R + 1)
        ox, oy = np.meshgrid(o, o)
        ix, iy = np.floor(xs).astype(np.int64), np.floor(ys).astype(np.int64)
        fx, fy = xs - ix - 0.5, ys - iy - 0.5
        dx = ox[None] - fx[:, None, None]
        dy = oy[None] - fy[:, None, None]
        wgt = np.exp(-(dx * dx + dy * dy) / (2.0 * sig[:, None, None] ** 2)) * b[:, None, None]
        px, py = ix[:, None, None] + ox[None], iy[:, None, None] + oy[None]
        ok = (px >= 0) & (px < W) & (py >= y0) & (py < y1) & (wgt > 0.003)
        flat = py * W + px
        grp = np.broadcast_to(rng.integers(0, groups, n)[:, None, None], ok.shape)
        self.groups = []
        for g in range(groups):
            sel = ok & (grp == g)
            idx = flat[sel]
            vals = wgt[sel][:, None] * np.broadcast_to(cols[:, None, None, :], ok.shape + (3,))[sel]
            if visible is not None:
                vals = vals * visible.reshape(-1)[idx][:, None]
            u, inv = np.unique(idx, return_inverse=True)
            acc = np.zeros((u.size, 3), np.float64)
            np.add.at(acc, inv, vals)
            k = int(rng.integers(2, 10))
            self.groups.append((u, acc.astype(F32), k, float(rng.uniform(0, TAU))))
        self.twinkle = twinkle

    def add(self, img, p, scale=1.0):
        flat = img.reshape(-1, 3)
        for idx, val, k, ph in self.groups:
            flat[idx] += val * (scale * (1.0 + self.twinkle * math.sin(k * p + ph)))


def splat(img, x, y, sigma, color, amount, reach=3.0):
    """Additive gaussian dot at float position (x, y)."""
    H, W = img.shape[:2]
    r = int(math.ceil(sigma * reach))
    x0, x1 = max(int(x) - r, 0), min(int(x) + r + 2, W)
    y0, y1 = max(int(y) - r, 0), min(int(y) + r + 2, H)
    if x0 >= x1 or y0 >= y1:
        return
    gx = np.arange(x0, x1, dtype=F32) + 0.5 - x
    gy = np.arange(y0, y1, dtype=F32)[:, None] + 0.5 - y
    w = np.exp(-(gx * gx + gy * gy) / (2.0 * sigma * sigma)) * amount
    img[y0:y1, x0:x1] += w[..., None] * color


# --------------------------------------------------------------------------
# scenes
# --------------------------------------------------------------------------
class Scene:
    key = ""
    title = ""
    orientation = "horizontal"
    blurb = ""

    def __init__(self, W, H, seconds, fps, seed=11):
        self.W, self.H, self.fps = W, H, fps
        self.nframes = int(round(seconds * fps))
        self.s = min(W, H) / 2160.0
        self.rng = np.random.default_rng(seed)
        self.setup()
        drng = np.random.default_rng(seed + 1)
        self.dither = (drng.random((H, W, 3), dtype=F32) - drng.random((H, W, 3), dtype=F32))

    def setup(self):
        pass

    def render(self, p, u):
        raise NotImplementedError

    def frame(self, i):
        u = (i % self.nframes) / self.nframes
        return self.render(TAU * u, u)

    def frame_u8(self, i):
        img = self.frame(i)
        img *= 255.0
        img += self.dither
        img += 0.5
        np.clip(img, 0.0, 255.0, out=img)
        return img.astype(np.uint8)


class AuroraLake(Scene):
    key, title, orientation = "aurora-lake", "Aurora-Lake", "horizontal"
    blurb = "Northern lights rippling over a mountain lake, mirrored in the water."

    def setup(self):
        W, H, s, rng = self.W, self.H, self.s, self.rng
        self.hz = hz = int(round(H * 0.62))
        v = (np.arange(hz, dtype=F32) + 0.5) / hz
        col = gradient([(0.0, hexrgb("#010309")), (0.55, hexrgb("#050f22")),
                        (0.85, hexrgb("#0a2136")), (1.0, hexrgb("#10304a"))], v)
        sky = np.repeat(col[:, None, :], W, axis=1)
        uu = (np.arange(W, dtype=F32) + 0.5) / W
        glow = np.exp(-((uu - 0.42) / 0.30) ** 2)[None, :] * smoothstep(0.45, 1.0, v)[:, None]
        sky += glow[..., None] * hexrgb("#0f3a3a") * 0.55
        self.sky = sky

        # far mountains, mid mountains, treeline on the far shore
        n1, n2 = noise1d(4096, rng, 1.7, lo=2), noise1d(4096, rng, 1.5, lo=2)
        far = H * (0.135 + 0.05 * sample1d(n1, uu * 0.45 + 0.1)
                   + 0.09 * np.exp(-((uu - 0.70) / 0.085) ** 2)
                   + 0.06 * np.exp(-((uu - 0.22) / 0.06) ** 2))
        far += H * 0.012 * np.abs(sample1d(noise1d(4096, rng, 0.9, lo=20), uu * 0.5))
        mid = H * (0.06 + 0.035 * sample1d(n2, uu * 0.5 + 0.3)
                   + 0.05 * np.exp(-((uu - 0.08) / 0.10) ** 2))
        y0 = int(hz - far.max()) - 4
        self.fy0 = y0
        cov_far = ridge_mask(hz - far, y0, hz)
        cov_mid = ridge_mask(hz - mid, y0, hz)

        def trees(d, ss):
            x = 0.0
            dens = noise1d(512, rng, 1.0, lo=1)
            while x < W:
                gap = W * (0.0025 + 0.004 * rng.random())
                x += gap
                if sample1d(dens, x / W * 0.7) < -0.45:
                    continue
                h = H * (0.018 + 0.03 * rng.random() ** 1.5) * (1.0 + 0.6 * max(0.0, float(sample1d(dens, x / W * 0.7))))
                pts = pine(x, hz + 2 * s, h, h * (0.32 + 0.12 * rng.random()), rng)
                d.polygon([((px) * ss, (py - y0) * ss) for px, py in pts], fill=255)
        cov_tree = ss_mask(W, hz - y0, 3, trees)

        rows = np.arange(y0, hz, dtype=F32)[:, None]
        depth_far = np.clip((rows - (hz - far)[None, :]) / (H * 0.12), 0, 1)
        snow_tex = resize(tile_noise(128, 512, rng, 1.1, sx=0.35), W, hz - y0)
        c_far = mixc(hexrgb("#17283a"), hexrgb("#0a1624"), depth_far) * (0.92 + 0.08 * np.clip(snow_tex, -2, 2))[..., None]
        rim = np.exp(-np.clip(rows - (hz - far)[None, :], 0, None) / (5.0 * s))
        c_far = c_far + rim[..., None] * hexrgb("#3b8f7a") * 0.25
        depth_mid = np.clip((rows - (hz - mid)[None, :]) / (H * 0.06), 0, 1)
        c_mid = mixc(hexrgb("#0b1621"), hexrgb("#050b13"), depth_mid)
        c_tree = hexrgb("#020407")

        acc = np.zeros((hz - y0, W, 3), F32)
        alpha = np.zeros((hz - y0, W), F32)
        for cov, c in ((cov_far, c_far), (cov_mid, c_mid), (cov_tree, c_tree)):
            acc = acc * (1.0 - cov[..., None]) + c * cov[..., None]
            alpha = alpha * (1.0 - cov) + cov
        self.fg, self.fg_a = acc.astype(F32), alpha.astype(F32)

        vis = np.ones((H, W), F32)
        vis[y0:hz] = 1.0 - alpha
        self.stars = Stars(W, H, int(5200 * W * H / 8294400), rng, s, 0, hz,
                           density=lambda x, y: 1.0 - smoothstep(0.25, 0.58, y), visible=vis)

        # aurora sampling grid (half width, quarter height)
        self.aw, self.ah = W // 2, hz // 4
        self.au = ((np.arange(self.aw) + 0.5) / self.aw).astype(np.float64)
        self.av = ((np.arange(self.ah) + 0.5) / self.ah).astype(F32)
        self.rays = [noise1d(4096, rng, 0.6, lo=30, hi=1400) for _ in range(2)]
        self.envs = [noise1d(256, rng, 1.2, lo=1) for _ in range(3)]
        R = lambda: float(rng.uniform(0, TAU))
        self.curtains = [
            dict(y0=0.60, h=0.34, amp=1.00, ph=R(), waves=[(0.075, 0.55, 1, R()), (0.035, 1.6, -1, R()), (0.014, 4.1, 2, R())]),
            dict(y0=0.44, h=0.25, amp=0.55, ph=R(), waves=[(0.060, 0.85, -1, R()), (0.028, 2.2, 1, R()), (0.012, 5.3, -2, R())]),
            dict(y0=0.74, h=0.22, amp=0.60, ph=R(), waves=[(0.050, 0.70, 1, R()), (0.022, 1.9, 2, R()), (0.010, 4.7, -1, R())]),
        ]

        # lake ripples
        Hl = H - hz
        dyn = np.arange(1, Hl + 1, dtype=F32)
        C = H * 0.2
        Z = (C / dyn)[:, None]
        X = ((np.arange(W, dtype=F32) - W / 2) / C)[None, :] * Z
        self.ripples = []
        for kz, kx, n, ax, ay in ((9.0, 1.1, 2, 4.0, 2.0), (17.0, -2.3, 3, 2.6, 1.4),
                                  (31.0, 3.7, 5, 1.6, 1.0), (55.0, -6.1, 7, 1.0, 0.6)):
            fade = np.clip(1.6 - (kz * C / dyn ** 2) / 0.55, 0.0, 1.0)
            env = (dyn / Hl) ** 0.85 * fade * s
            base = (kz * Z + kx * X + rng.uniform(0, TAU)).astype(F32)
            self.ripples.append((base, n, (env * ax)[:, None], (env * ay)[:, None]))
        self.dyn = dyn
        self.fres = (0.72 - 0.50 * (dyn / Hl) ** 0.6)[:, None, None].astype(F32)
        self.xcol = np.arange(W, dtype=F32)[None, :]
        self.deep = hexrgb("#01050a")

    def aurora(self, p):
        u, v = self.au, self.av[:, None]
        out = np.zeros((self.ah, self.aw, 3), F32)
        green, teal, violet, pink = hexrgb("#33ff8c"), hexrgb("#1ad9c0"), hexrgb("#7a3df0"), hexrgb("#ff4fa3")
        # rays fan out toward the zenith and waver a little along their length
        vv = self.av.astype(np.float64)[:, None]
        ur = u[None, :] + (u[None, :] - 0.5) * 0.18 * (1.0 - vv) + 0.0025 * np.sin(TAU * 1.5 * vv + p)
        r1 = sample1d(self.rays[0], ur * 0.45 + 0.004 * math.sin(p + 0.3))
        r2 = sample1d(self.rays[1], ur * 0.31 + 0.5 - 0.003 * math.sin(2 * p + 1.1))
        rays = np.clip(0.62 + 0.30 * (r1 + r2), 0.12, 1.6)
        for i, c in enumerate(self.curtains):
            base = c["y0"] + sum(A * np.sin(TAU * f * u + k * p + ph) for A, f, k, ph in c["waves"])
            hgt = c["h"] * (1.0 + 0.3 * np.sin(TAU * 1.3 * u + p + c["ph"]))
            env = np.clip(0.50 + 0.55 * sample1d(self.envs[i], u * 0.8 + 0.05 * math.sin(p + c["ph"])), 0, 1.2) ** 1.8
            env = env * (0.78 + 0.22 * np.sin(TAU * u - p + c["ph"]))
            d = (base.astype(F32)[None, :] - v) / hgt.astype(F32)[None, :]
            up = np.exp(-np.maximum(d, 0.0) * 2.5)
            low = np.exp(-(np.minimum(d, 0.0) / 0.045) ** 2)
            prof = np.where(d >= 0, up, low) * (c["amp"] * env.astype(F32))[None, :]
            prof = prof * (0.65 + 0.35 * rays + 0.35 * (rays - 1.0) * smoothstep(0.0, 0.6, d)) * smoothstep(0.0, 0.3, v)
            col = mixc(green, teal, smoothstep(0.05, 0.55, d))
            col = mixc(col, violet, smoothstep(0.45, 1.35, d)) * (1.0 - 0.45 * smoothstep(0.6, 1.5, d))[..., None]
            col = col + pink * (0.5 * np.exp(-(np.minimum(d, 0.0) / 0.03) ** 2) * (d < 0.02))[..., None]
            out += col * prof[..., None]
        return out

    def render(self, p, u):
        W, H, hz, y0 = self.W, self.H, self.hz, self.fy0
        img = np.empty((H, W, 3), F32)
        above = img[:hz]
        above[...] = self.sky
        self.stars.add(img, p)
        above += np.maximum(resize(self.aurora(p) * 0.85, W, hz), 0.0)
        reg = above[y0:hz]
        reg *= (1.0 - self.fg_a)[..., None]
        reg += self.fg
        np.clip(above, 0.0, 4.0, out=above)

        dx = np.zeros((H - hz, W), F32)
        dy = np.zeros((H - hz, W), F32)
        for base, n, ax, ay in self.ripples:
            c = np.cos(base + F32(n * p))
            dx += ax * c
            dy += ay * c
        src_r = np.clip(hz - self.dyn[:, None] + dy, 0, hz - 1).astype(np.int32)
        src_c = np.clip(self.xcol + dx, 0, W - 1).astype(np.int32)
        refl = above[src_r, src_c]
        img[hz:] = refl * self.fres + self.deep * (1.0 - self.fres)
        return img


class NeonHorizon(Scene):
    key, title, orientation = "neon-horizon", "Neon-Horizon", "horizontal"
    blurb = "Retro sunset over an endless scrolling neon grid."

    def setup(self):
        W, H, s, rng = self.W, self.H, self.s, self.rng
        self.hz = hz = int(round(H * 0.60))
        v = (np.arange(hz, dtype=F32) + 0.5) / hz
        col = gradient([(0.0, hexrgb("#05010f")), (0.35, hexrgb("#170536")), (0.62, hexrgb("#3d0a5c")),
                        (0.84, hexrgb("#8a1667")), (0.96, hexrgb("#e2456a")), (1.0, hexrgb("#ff7a59"))], v)
        sky = np.repeat(col[:, None, :], W, axis=1)
        X = np.arange(W, dtype=F32)[None, :] + 0.5
        Y = np.arange(hz, dtype=F32)[:, None] + 0.5
        self.cx, self.cy, self.r = W * 0.5, hz - H * 0.115, H * 0.215
        dist = np.sqrt((X - self.cx) ** 2 + (Y - self.cy) ** 2)
        sky += (np.exp(-dist / (H * 0.33)) * 0.30)[..., None] * hexrgb("#ff2f8a")
        sky += (np.exp(-dist / (H * 0.13)) * 0.22)[..., None] * hexrgb("#ffb35c")
        self.sky = sky

        # sun box
        r = self.r
        self.sy0, self.sy1 = max(int(self.cy - r) - 2, 0), hz
        self.sx0, self.sx1 = int(self.cx - r) - 2, int(self.cx + r) + 3
        sy = np.arange(self.sy0, self.sy1, dtype=F32)[:, None] + 0.5
        sx = np.arange(self.sx0, self.sx1, dtype=F32)[None, :] + 0.5
        d = np.sqrt((sx - self.cx) ** 2 + (sy - self.cy) ** 2)
        self.sun_cov = np.clip(r - d + 0.5, 0, 1)
        self.sun_v = ((sy - self.cy) / r).astype(F32)               # -1 top .. 1 bottom
        self.sun_col = gradient([(-1.0, hexrgb("#fff27a")), (-0.2, hexrgb("#ffb347")),
                                 (0.4, hexrgb("#ff5a6e")), (1.0, hexrgb("#ff2a9a"))],
                                np.broadcast_to(self.sun_v, self.sun_cov.shape)) * 1.08

        # side mountains with a neon rim
        uu = (np.arange(W, dtype=F32) + 0.5) / W
        side = smoothstep(0.13, 0.42, np.abs(uu - 0.5))
        n1 = noise1d(4096, rng, 1.6, lo=3)
        ridged = 1.0 - np.abs(sample1d(n1, uu * 0.5))
        jag = sample1d(noise1d(4096, rng, 1.0, lo=30), uu * 0.6)
        mh = H * (0.006 + 0.16 * side * (0.25 + 0.75 * ridged ** 2 + 0.03 * jag))
        mh = np.maximum(mh, H * 0.004)
        self.my0 = my0 = int(hz - mh.max()) - 6
        cov = ridge_mask(hz - mh, my0, hz)
        rows = np.arange(my0, hz, dtype=F32)[:, None]
        depth = np.clip((rows - (hz - mh)[None, :]) / (H * 0.18), 0, 1)
        mcol = mixc(hexrgb("#2a0b52"), hexrgb("#0d0322"), depth ** 0.6)
        edge = np.exp(-np.abs(rows + 0.5 - (hz - mh)[None, :]) / (1.6 * s))
        glow = np.exp(-np.clip(rows - (hz - mh)[None, :], 0, None) / (14 * s))
        self.m_cov = cov
        self.m_col = (mcol * cov[..., None] + edge[..., None] * hexrgb("#ff4fd8") * 0.9
                      + (glow * cov)[..., None] * hexrgb("#a12bd9") * 0.20).astype(F32)
        vis = np.ones((H, W), F32)
        vis[my0:hz] = 1.0 - cov
        self.stars = Stars(W, H, int(2600 * W * H / 8294400), rng, s, 0, hz,
                           density=lambda x, y: 1.0 - smoothstep(0.15, 0.48, y), visible=vis, bright=0.9)

        # ground grid
        Hg = H - hz
        dyr = np.arange(Hg, dtype=F32) + 0.5
        zb = 0.85
        Cz = zb * Hg
        self.z = Cz / dyr
        self.fpz = Cz / dyr ** 2
        Fx = zb * W / 15.0
        xs = (np.arange(W, dtype=F32) + 0.5 - W / 2)[None, :]
        Xw = xs * self.z[:, None] / Fx
        gx = self.z[:, None] / Fx
        gy = xs / Fx * self.fpz[:, None]
        fpx = np.sqrt(gx * gx + gy * gy)
        self.lw = 0.012
        self.vlines = self.coverage(np.abs(Xw - np.round(Xw)), fpx, self.lw)
        self.fade = (np.exp(-self.z / 28.0) * smoothstep(0.0, 6.0 * s, dyr))[:, None].astype(F32)
        vglow = resize(blur(resize(self.vlines * self.fade, W // 4, Hg // 4), 5.0), W, Hg)
        self.vglow = np.clip(vglow, 0, None)
        yy = dyr[:, None]
        base = gradient([(0.0, hexrgb("#3a0b4f")), (0.18, hexrgb("#1a0530")), (1.0, hexrgb("#07020f"))],
                        np.broadcast_to(dyr / Hg, (Hg,)))
        ground = np.repeat(base[:, None, :], W, axis=1)
        refl = np.exp(-((xs - 0) / (W * 0.10)) ** 2) * np.exp(-yy / (Hg * 0.30))
        ground += refl[..., None] * hexrgb("#ff4f9a") * 0.28
        self.ground = ground.astype(F32)
        hl = np.arange(H, dtype=F32)[:, None] + 0.5 - hz
        self.hline = (np.exp(-np.abs(hl) / (2.2 * s)) * 0.9 + np.exp(-np.abs(hl) / (60 * s)) * 0.18).astype(F32)
        self.neon = hexrgb("#ff3cf0")
        self.cyan = hexrgb("#53e8ff")
        k = np.arange(-int(24 * s) - 1, int(24 * s) + 2, dtype=F32)
        self.k1 = np.exp(-(k / (8.0 * s)) ** 2)
        self.k1 /= self.k1.sum()

    @staticmethod
    def coverage(dist, fp, w):
        hw = np.maximum(w / fp, 0.65)
        energy = np.minimum(w / fp / 0.65, 1.0)
        c = np.clip(hw + 0.5 - dist / fp, 0.0, 1.0) * energy
        avg = 2.0 * w
        lod = smoothstep(0.12, 0.35, fp)
        return (c * (1.0 - lod) + avg * lod).astype(F32)

    def render(self, p, u):
        W, H, hz = self.W, self.H, self.hz
        img = np.empty((H, W, 3), F32)
        img[:hz] = self.sky
        self.stars.add(img, p)

        # sun with downward-scrolling cut-outs
        sv = self.sun_v
        q = sv * 6.0 - u * 2.0
        fr = q - np.floor(q)
        g = 0.06 + 0.46 * np.clip((sv + 0.12) / 1.1, 0, 1)
        e = np.where(fr < g, np.minimum(fr, g - fr), -np.minimum(fr - g, 1.0 - fr))
        gap = np.clip(e * (self.r / 6.0) + 0.5, 0, 1) * (sv > -0.12)
        a = (self.sun_cov * (1.0 - gap))[..., None]
        box = img[self.sy0:self.sy1, self.sx0:self.sx1]
        box *= 1.0 - a
        box += self.sun_col * a

        reg = img[self.my0:hz]
        reg *= (1.0 - self.m_cov)[..., None]
        reg += self.m_col

        # ground
        z = self.z + u * 12.0
        hlines = self.coverage(np.abs(z - np.round(z)), self.fpz, self.lw * 1.4)
        hg = np.convolve(hlines * self.fade[:, 0], self.k1, mode="same")[:, None]
        grid = 1.0 - (1.0 - self.vlines) * (1.0 - hlines[:, None])
        neon = (grid * self.fade)[..., None] * self.neon * 1.25 + ((self.vglow + hg) * 0.55)[..., None] * self.neon
        neon += (grid * self.fade)[..., None] ** 3 * 0.25
        img[hz:] = self.ground + neon
        img += self.hline[..., None] * hexrgb("#ff5ad1") * 0.8
        return img


class MoonlitSea(Scene):
    key, title, orientation = "moonlit-sea", "Moonlit-Sea", "horizontal"
    blurb = "Rolling night ocean with a glittering moon path and a lighthouse."

    def setup(self):
        W, H, s, rng = self.W, self.H, self.s, self.rng
        self.hz = hz = int(round(H * 0.56))
        self.Fp = Fp = W * 0.80
        cx = W / 2
        el, az = math.radians(10.5), math.radians(11.0)
        L = np.array([math.sin(az) * math.cos(el), math.sin(el), math.cos(az) * math.cos(el)], F32)
        self.L = L
        mx, my = cx + Fp * L[0] / L[2], hz - Fp * L[1] / L[2]
        X = np.arange(W, dtype=F32)[None, :] + 0.5
        Y = np.arange(hz, dtype=F32)[:, None] + 0.5
        d = np.stack(np.broadcast_arrays((X - cx) / Fp, (hz - Y) / Fp, np.ones_like(X)), -1)
        d /= np.linalg.norm(d, axis=-1, keepdims=True)
        cosm = d @ L
        ey = d[..., 1]
        sky = gradient([(0.0, hexrgb("#24395a")), (0.05, hexrgb("#142844")), (0.22, hexrgb("#081428")),
                        (0.6, hexrgb("#02060f"))], ey)
        sky += (np.exp(-(1 - cosm) / 0.0035) * 0.30)[..., None] * hexrgb("#9fb8d9")
        sky += (np.exp(-(1 - cosm) / 0.04) * 0.10)[..., None] * hexrgb("#5e7fa8")

        # static wisps of cloud, lit near the moon
        ct = tile_noise(128, 512, rng, 1.7, sx=0.18)
        cl = resize(ct, W, hz)
        band = np.exp(-((Y / hz - 0.30) / 0.10) ** 2) * 0.7 + np.exp(-((Y / hz - 0.80) / 0.06) ** 2)
        cloud = smoothstep(0.2, 1.8, cl) * band
        lit = 0.25 + 0.9 * np.exp(-(1 - cosm) / 0.02)
        sky = sky * (1.0 - 0.30 * cloud[..., None]) + (cloud * lit)[..., None] * hexrgb("#7d8fae") * 0.16

        # moon disk
        mr = H * 0.034
        dm = np.sqrt((X - mx) ** 2 + (Y - my) ** 2)
        cov = np.clip(mr - dm + 0.5, 0, 1)
        maria = resize(tile_noise(64, 64, rng, 1.4), int(mr * 2) + 1, int(mr * 2) + 1)
        tex = np.zeros((hz, W), F32)
        y0m, x0m = int(my - mr), int(mx - mr)
        tex[y0m:y0m + maria.shape[0], x0m:x0m + maria.shape[1]] = maria
        limb = np.sqrt(np.clip(1 - (dm / mr) ** 2, 0, 1))
        mcol = (0.82 + 0.10 * np.clip(tex, -1.5, 1.5)) * (0.75 + 0.25 * limb)
        sky = sky * (1 - cov[..., None]) + (cov * mcol)[..., None] * hexrgb("#f6f1e2") * 1.15

        # island with a lighthouse on the left horizon
        uu = (np.arange(W, dtype=F32) + 0.5) / W
        ih = H * (0.028 * np.exp(-((uu - 0.17) / 0.07) ** 2) + 0.012 * np.exp(-((uu - 0.27) / 0.05) ** 2))
        ih *= 1.0 + 0.25 * sample1d(noise1d(2048, rng, 1.0, lo=8), uu * 0.5)
        ih = np.where(ih < 1.0, 0.0, ih)
        self.lx, self.ltop = W * 0.165, hz - H * 0.028 - H * 0.032
        def tower(dr, ss):
            bx, by, th, tw = self.lx, hz - H * 0.025, H * 0.032, H * 0.006
            dr.polygon([((bx - tw) * ss, by * ss), ((bx - tw * 0.6) * ss, (by - th) * ss),
                        ((bx + tw * 0.6) * ss, (by - th) * ss), ((bx + tw) * ss, by * ss)], fill=255)
        tcov = ss_mask(W, hz, 3, tower)
        icov = np.maximum(ridge_mask(hz - ih, 0, hz), tcov)
        sky = sky * (1 - icov[..., None]) + icov[..., None] * hexrgb("#03060c")
        vis = (1.0 - cov) * (1.0 - icov) * (1.0 - smoothstep(0.3, 0.9, cloud))
        visfull = np.ones((H, W), F32)
        visfull[:hz] = vis
        self.sky = sky.astype(F32)
        self.stars = Stars(W, H, int(3400 * W * H / 8294400), rng, s, 0, hz,
                           density=lambda x, y: (1.0 - smoothstep(0.2, 0.5, y)), visible=visfull)

        # ocean geometry
        Hs = H - hz
        dy = np.arange(Hs, dtype=F32) + 0.5
        h_cam = 3.0
        Z = (h_cam * Fp / dy)[:, None]
        Xw = (np.arange(W, dtype=F32)[None, :] + 0.5 - cx) / Fp * Z
        D = np.stack(np.broadcast_arrays((np.arange(W, dtype=F32)[None, :] + 0.5 - cx) / Fp,
                                         -dy[:, None] / Fp, np.ones((Hs, W), F32)), -1)
        self.D = (D / np.linalg.norm(D, axis=-1, keepdims=True)).astype(F32)
        fp = np.maximum(Z / Fp, Z * Z / (h_cam * Fp))
        g, T = 9.81, self.nframes / self.fps
        self.waves = []
        var_lost = np.zeros_like(fp)
        for n, steep in ((4, 0.028), (6, 0.032), (9, 0.036), (13, 0.04), (18, 0.042),
                         (25, 0.042), (34, 0.04), (46, 0.036), (62, 0.03)):
            k = (TAU * n / T) ** 2 / g
            th = math.radians(rng.uniform(-38, 38))
            kx, kz = k * math.sin(th), -k * math.cos(th)
            a = steep / k
            f = np.clip(1.0 - (fp * k - 0.35) / 1.1, 0.0, 1.0)
            var_lost += 0.5 * (a * k) ** 2 * (1.0 - f * f)
            base = (kx * Xw + kz * Z + rng.uniform(0, TAU)).astype(F32)
            self.waves.append((base, n, (a * kx * f).astype(F32), (a * kz * f).astype(F32)))
        self.V0 = 0.0022
        self.V = (self.V0 + 4.0 * var_lost).astype(F32)
        self.cx, self.Hs = cx, Hs
        self.deep = hexrgb("#010409")

    def render(self, p, u):
        W, H, hz, Fp = self.W, self.H, self.hz, self.Fp
        img = np.empty((H, W, 3), F32)
        img[:hz] = self.sky
        self.stars.add(img, p)
        blink = 0.5 + 0.5 * math.cos(4 * p)
        splat(img, self.lx, self.ltop - 2 * self.s, 3.0 * self.s, hexrgb("#ffe6a8"), 2.5 * blink ** 6 + 0.25)
        splat(img, self.lx, self.ltop - 2 * self.s, 26.0 * self.s, hexrgb("#ffc26b"), 0.35 * blink ** 6)

        sx = np.zeros((self.Hs, W), F32)
        sz = np.zeros((self.Hs, W), F32)
        for base, n, akx, akz in self.waves:
            sn = np.sin(base - F32(n * p))
            sx -= akx * sn
            sz -= akz * sn
        N = np.stack([-sx, np.ones_like(sx), -sz], -1)
        N /= np.linalg.norm(N, axis=-1, keepdims=True)
        D = self.D
        dn = np.einsum("ijk,ijk->ij", D, N)
        R = D - 2.0 * dn[..., None] * N
        cos_i = np.clip(-dn, 0.0, 1.0)
        fres = 0.02 + 0.98 * (1.0 - cos_i) ** 5
        rz = np.maximum(R[..., 2], 1e-3)
        px = np.clip(self.cx + Fp * R[..., 0] / rz, 0, W - 1).astype(np.int32)
        py = np.clip(hz - Fp * R[..., 1] / rz, 0, hz - 1).astype(np.int32)
        refl = img[py, px]
        rl = R @ self.L
        spec = np.exp(-(1.0 - rl) / self.V) * (self.V0 / self.V) * 5.0
        water = refl * fres[..., None] + (spec * fres)[..., None] * hexrgb("#fff4dc")
        water += self.deep * (1.0 - fres[..., None])
        img[hz:] = water
        return img


class LavaLamp(Scene):
    key, title, orientation = "lava-lamp", "Lava-Lamp", "vertical"
    blurb = "Glossy molten wax blobs rising and falling."

    def setup(self):
        W, H, rng = self.W, self.H, self.rng
        self.lw, self.lh = W // 2, H // 2
        self.lx = ((np.arange(self.lw) + 0.5) * (W / self.lw)).astype(F32)
        self.ly = ((np.arange(self.lh) + 0.5) * (H / self.lh)).astype(F32)
        self.blobs = []
        for i in range(11):
            r = W * rng.uniform(0.05, 0.105)
            self.blobs.append(dict(
                r2=F32(r * r), X=W * rng.uniform(0.18, 0.82), ax=W * rng.uniform(0.03, 0.10),
                kx=int(rng.integers(1, 3)), px=rng.uniform(0, TAU),
                Y=H * rng.uniform(0.40, 0.60), ay=H * rng.uniform(0.18, 0.40),
                ky=1 if i % 4 else 2, py=rng.uniform(0, TAU)))
        yy = self.ly
        self.pool = ((W * 0.17) ** 2 / ((H + W * 0.06 - yy) ** 2)).astype(F32) \
            + ((W * 0.11) ** 2 / ((yy + W * 0.05) ** 2)).astype(F32)
        v = (np.arange(H, dtype=F32) + 0.5) / H
        uu = (np.arange(W, dtype=F32) + 0.5) / W
        bg = gradient([(0.0, hexrgb("#14031f")), (0.5, hexrgb("#260636")), (1.0, hexrgb("#3d0a2c"))], v)
        vig = 1.0 - 0.45 * ((uu[None, :] - 0.5) * 2) ** 2
        self.bg = (np.repeat(bg[:, None, :], W, axis=1) * vig[..., None]).astype(F32)
        self.ramp = gradient([(0.0, hexrgb("#ff3df5")), (0.45, hexrgb("#ff3d7a")),
                              (0.8, hexrgb("#ff6a2a")), (1.0, hexrgb("#ffb03a"))], v)[:, None, :]
        Lv = np.array([-0.45, -0.6, 0.66], F32)
        self.Lv = Lv / np.linalg.norm(Lv)
        Hv = self.Lv + np.array([0, 0, 1], F32)
        self.Hv = Hv / np.linalg.norm(Hv)
        self.R0 = F32(W * 0.11)

    def render(self, p, u):
        W, H = self.W, self.H
        fld = np.repeat(self.pool[:, None], self.lw, axis=1)
        for b in self.blobs:
            cx = b["X"] + b["ax"] * math.sin(b["kx"] * p + b["px"])
            cy = b["Y"] - b["ay"] * math.cos(b["ky"] * p + b["py"])
            d2 = ((self.ly - cy) ** 2)[:, None] + ((self.lx - cx) ** 2)[None, :]
            fld += b["r2"] / (d2 + 1.0)
        F = np.maximum(resize(fld, W, H), 1e-4)
        hgt = np.sqrt(np.clip(1.0 - 1.0 / F, 0.0, 1.0))
        gy, gx = np.gradient(F)
        cov = np.clip((F - 1.0) / (np.sqrt(gx * gx + gy * gy) + 1e-6) + 0.5, 0.0, 1.0)
        hy, hx = np.gradient(hgt)
        N = np.stack([-hx * self.R0, -hy * self.R0, np.ones_like(hx)], -1)
        N /= np.linalg.norm(N, axis=-1, keepdims=True)
        diff = np.clip(N @ self.Lv, 0, 1)
        spec = np.clip(N @ self.Hv, 0, 1) ** 60
        rim = (1.0 - N[..., 2]) ** 2
        body = self.ramp * (0.45 + 0.75 * diff)[..., None] + (spec * 0.9)[..., None] \
            + rim[..., None] * hexrgb("#ffd08a") * 0.35
        halo = np.clip(F, 0, 1) ** 3 * 0.30
        img = self.bg + self.ramp * halo[..., None]
        img += (body - img) * cov[..., None]
        return img


class FireflyForest(Scene):
    key, title, orientation = "firefly-forest", "Firefly-Forest", "vertical"
    blurb = "Misty pine forest at night with drifting, blinking fireflies."

    def setup(self):
        W, H, s, rng = self.W, self.H, self.s, self.rng
        v = (np.arange(H, dtype=F32) + 0.5) / H
        X = np.arange(W, dtype=F32)[None, :] + 0.5
        Y = np.arange(H, dtype=F32)[:, None] + 0.5
        sky = gradient([(0.0, hexrgb("#01040c")), (0.30, hexrgb("#071a2b")), (0.55, hexrgb("#15384a")),
                        (0.75, hexrgb("#2c5a5f"))], v)
        sky = np.repeat(sky[:, None, :], W, axis=1)
        mx, my, mr = W * 0.72, H * 0.12, W * 0.055
        dm = np.sqrt((X - mx) ** 2 + (Y - my) ** 2)
        sky += (np.exp(-dm / (W * 0.25)) * 0.22)[..., None] * hexrgb("#7fb3c4")
        mcov = np.clip(mr - dm + 0.5, 0, 1)
        sky = sky * (1 - mcov[..., None]) + mcov[..., None] * hexrgb("#e9f2e8")

        def layer(count, base0, base1, hmin, hmax, wscale, edge_bias=0.0):
            def draw(d, ss):
                xs = np.sort(rng.uniform(-0.05, 1.05, count))
                for x in xs:
                    side = abs(x - 0.5) * 2
                    h = H * rng.uniform(hmin, hmax) * (1.0 + edge_bias * side)
                    base = H * rng.uniform(base0, base1)
                    pts = pine(x * W, base, h, h * wscale * rng.uniform(0.8, 1.2), rng)
                    d.polygon([(px * ss, py * ss) for px, py in pts], fill=255)
                gx = np.linspace(0, W, 64)
                gy = H * base1 + H * 0.006 * np.sin(gx / W * TAU * rng.uniform(1, 3) + rng.uniform(0, TAU))
                d.polygon([(a * ss, b * ss) for a, b in zip(gx, gy)] + [(W * ss, H * ss), (0, H * ss)], fill=255)
            return ss_mask(W, H, 2, draw)

        L0 = layer(22, 0.70, 0.74, 0.22, 0.34, 0.30)
        L1 = layer(16, 0.78, 0.83, 0.30, 0.45, 0.30)
        L2 = layer(10, 0.88, 0.93, 0.42, 0.60, 0.28, 0.3)

        def near(d, ss):
            for x, h, w in ((0.03, 0.92, 0.55), (0.97, 0.80, 0.50)):
                pts = pine(x * W, H * 1.02, H * h, W * w, rng)
                d.polygon([(px * ss, py * ss) for px, py in pts], fill=255)
            gy = H * 0.965
            d.rectangle([0, gy * ss, W * ss, H * ss], fill=255)
            for _ in range(int(900 * s) + 200):
                gx = rng.uniform(0, W)
                gh = H * rng.uniform(0.01, 0.05)
                lean = rng.uniform(-0.4, 0.4) * gh
                bw = rng.uniform(2, 6) * s
                d.polygon([((gx - bw) * ss, (gy + 4) * ss), ((gx + lean) * ss, (gy - gh) * ss),
                           ((gx + bw) * ss, (gy + 4) * ss)], fill=255)
        L3 = ss_mask(W, H, 2, near)

        fogc = hexrgb("#3d6e70")
        fogged = lambda c, top, bot, amt: mixc(hexrgb(c), fogc, smoothstep(top, bot, v) * amt)[:, None, :]
        base = sky.copy()
        for cov, col in ((L0, fogged("#1f4a52", 0.45, 0.76, 0.45)), (L1, fogged("#11303a", 0.50, 0.85, 0.35))):
            base = base * (1 - cov[..., None]) + col * cov[..., None]
        total = 1 - (1 - L0) * (1 - L1) * (1 - L2) * (1 - L3)
        vis = (1.0 - total) * (1.0 - mcov)
        self.stars = Stars(W, H, int(3000 * W * H / 8294400), rng, s, 0, int(H * 0.7),
                           density=lambda x, y: 1.0 - smoothstep(0.25, 0.6, y), visible=vis)
        self.base = base.astype(F32)
        self.L2, self.L2c = L2, fogged("#07161d", 0.55, 0.95, 0.22)
        self.L3, self.L3c = L3, hexrgb("#010406")
        self.l2y0 = int(np.argmax(L2.max(axis=1) > 0))
        self.l3y0 = int(np.argmax(L3.max(axis=1) > 0))

        # mist: two tileable textures scrolled one tile per loop in opposite directions
        self.mw, self.mh = W // 4, H // 4
        self.mA = tile_noise(self.mh, self.mw, rng, 1.3, sx=0.35)
        self.mB = tile_noise(self.mh, self.mw, rng, 1.1, sx=0.5)
        mv = (np.arange(self.mh, dtype=F32) + 0.5) / self.mh
        self.mband = (np.exp(-((mv - 0.80) / 0.07) ** 2) * 0.55 + np.exp(-((mv - 0.90) / 0.05) ** 2) * 0.45)[:, None]
        self.fogc = fogc

        def flies(n, y0, y1, core, halo, amt):
            out = []
            for _ in range(n):
                out.append(dict(X=rng.uniform(0.04, 0.96) * W, Y=rng.uniform(y0, y1) * H,
                                ax=rng.uniform(40, 200) * s, ay=rng.uniform(30, 140) * s,
                                kx=int(rng.integers(1, 4)), ky=int(rng.integers(1, 4)),
                                px=rng.uniform(0, TAU), py=rng.uniform(0, TAU),
                                kb=int(rng.integers(3, 8)), pb=rng.uniform(0, TAU),
                                core=core * rng.uniform(0.8, 1.25), halo=halo * rng.uniform(0.8, 1.3),
                                amt=amt * rng.uniform(0.6, 1.0)))
            return out
        self.far = flies(60, 0.55, 0.93, 3.0 * s, 18 * s, 0.8)
        self.near = flies(28, 0.48, 0.97, 5.5 * s, 42 * s, 1.0)
        self.c_core, self.c_halo = hexrgb("#f8ffb8"), hexrgb("#b6ff4a")

    def flies(self, img, group, p):
        for f in group:
            x = f["X"] + f["ax"] * math.sin(f["kx"] * p + f["px"]) + 0.3 * f["ax"] * math.sin(2 * f["kx"] * p + f["py"])
            y = f["Y"] + f["ay"] * math.sin(f["ky"] * p + f["py"])
            b = 0.5 + 0.5 * math.sin(f["kb"] * p + f["pb"])
            b = 0.06 + 0.94 * (b * b * (3 - 2 * b)) ** 3
            splat(img, x, y, f["core"], self.c_core, 1.2 * b * f["amt"])
            splat(img, x, y, f["halo"], self.c_halo, 0.28 * b * f["amt"])

    def render(self, p, u):
        W, H = self.W, self.H
        img = self.base.copy()
        self.stars.add(img, p)
        m = 0.6 * shift_tile(self.mA, u * self.mw) + 0.4 * shift_tile(self.mB, -u * self.mw)
        mist = resize(smoothstep(-0.6, 1.6, m) * self.mband, W, H)
        np.clip(mist, 0, 1, out=mist)
        img += (self.fogc - img) * (mist * 0.55)[..., None]
        self.flies(img, self.far, p)
        y0 = self.l2y0
        a = self.L2[y0:, :, None]
        img[y0:] += (self.L2c[y0:] - img[y0:]) * a
        self.flies(img, self.near, p)
        y0 = self.l3y0
        a = self.L3[y0:, :, None]
        img[y0:] += (self.L3c - img[y0:]) * a
        return img


class SunlitDepths(Scene):
    key, title, orientation = "sunlit-depths", "Sunlit-Depths", "vertical"
    blurb = "Sun shafts swaying through deep blue water, drifting particles and bubbles."

    def setup(self):
        W, H, s, rng = self.W, self.H, self.s, self.rng
        v = (np.arange(H, dtype=F32) + 0.5) / H
        uu = (np.arange(W, dtype=F32) + 0.5) / W
        bg = gradient([(0.0, hexrgb("#3fd0d8")), (0.08, hexrgb("#1593ad")), (0.3, hexrgb("#08597a")),
                       (0.6, hexrgb("#04304e")), (0.85, hexrgb("#021628")), (1.0, hexrgb("#010a14"))], v)
        vig = 1.0 - 0.35 * ((uu - 0.5) * 2) ** 2
        self.bg = (np.repeat(bg[:, None, :], W, axis=1) * vig[None, :, None]).astype(F32)
        self.rw, self.rh = W // 4, H // 4
        rx = (np.arange(self.rw, dtype=F32) + 0.5) * 4
        ry = (np.arange(self.rh, dtype=F32) + 0.5) * 4
        x0, y0 = W * 0.58, -H * 0.30
        self.theta = np.arctan2(rx[None, :] - x0, ry[:, None] - y0).astype(np.float64)
        dist = np.hypot(rx[None, :] - x0, ry[:, None] - y0)
        self.rfall = (np.exp(-(dist - H * 0.3) / (H * 0.42)) * smoothstep(0.0, 0.06, ry / H)[:, None]).astype(F32)
        self.rt = [noise1d(1024, rng, 0.8, lo=6, hi=160) for _ in range(3)]
        # surface caustics band
        self.cw, self.ch = W // 4, max(int(H * 0.09) // 4, 8)
        self.cA = tile_noise(self.ch, self.cw, rng, 1.6, lo=3)
        self.cB = tile_noise(self.ch, self.cw, rng, 1.6, lo=3)
        cv = (np.arange(self.ch, dtype=F32) + 0.5) / self.ch
        self.cband = (1.0 - smoothstep(0.0, 1.0, cv))[:, None] ** 1.5

        # marine snow: each particle lives exactly one loop/k, fading in and out
        n = int(1700 * W * H / 8294400)
        self.ms = dict(x=rng.uniform(0, W, n), y=rng.uniform(-0.02 * H, H, n),
                       dx=rng.uniform(-40, 40, n) * s, dy=rng.uniform(60, 220, n) * s,
                       k=rng.integers(1, 3, n), ph=rng.random(n),
                       sw=rng.uniform(4, 18, n) * s, swk=rng.integers(1, 4, n), swp=rng.uniform(0, TAU, n),
                       b=rng.uniform(0.25, 1.0, n) ** 2)
        o = np.arange(-3, 4)
        self.ox, self.oy = np.meshgrid(o, o)
        self.ox = self.ox.reshape(-1)
        self.oy = self.oy.reshape(-1)

        # bubble streams
        self.bubbles = []
        for sx in (0.16, 0.83, 0.44):
            for j in range(int(rng.integers(14, 22))):
                self.bubbles.append(dict(X=W * sx + rng.normal(0, 18 * s), r=rng.uniform(4, 16) * s,
                                         k=int(rng.integers(1, 3)), ph=rng.random(),
                                         wa=rng.uniform(6, 22) * s, wk=int(rng.integers(3, 8)), wp=rng.uniform(0, TAU)))

    def render(self, p, u):
        W, H, s = self.W, self.H, self.s
        img = self.bg.copy()
        th = self.theta
        r1 = sample1d(self.rt[0], th * 0.30 + 0.010 * math.sin(p + 0.4))
        r2 = sample1d(self.rt[1], th * 0.47 + 0.3 - 0.008 * math.sin(2 * p + 1.3))
        r3 = sample1d(self.rt[2], th * 0.21 + 0.6 + 0.006 * math.sin(3 * p + 2.1))
        rays = np.clip(0.35 * r1 + 0.35 * r2 + 0.3 * r3, 0, None) ** 1.6
        rays *= self.rfall * (1.0 + 0.10 * math.sin(p))
        rays_full = np.clip(resize(rays, W, H), 0, None)
        img += rays_full[..., None] * hexrgb("#a7f6ff") * 0.30

        ca = shift_tile(self.cA, u * self.cw, 0.0)
        cb = shift_tile(self.cB, -u * self.cw, 0.0)
        caus = (0.35 + 0.65 * np.clip(1.0 - np.abs(ca + cb) * 0.5, 0, 1) ** 4) * self.cband
        band = np.maximum(resize(caus, W, self.ch * 4), 0.0)
        img[:band.shape[0]] += band[..., None] * hexrgb("#d9ffff") * 0.25

        m = self.ms
        life = (u * m["k"] + m["ph"]) % 1.0
        alpha = np.sin(np.pi * life) ** 2 * m["b"]
        x = m["x"] + m["dx"] * life + m["sw"] * np.sin(m["swk"] * p + m["swp"])
        y = m["y"] + m["dy"] * life
        ix, iy = np.floor(x).astype(np.int64), np.floor(y).astype(np.int64)
        lit = 0.35 + 2.2 * rays_full[np.clip(iy, 0, H - 1), np.clip(ix, 0, W - 1)]
        amp = alpha * lit * 0.55
        sig = (0.7 + 0.9 * m["b"]) * max(s, 0.4)
        ddx = self.ox[None, :] + 0.5 - (x - ix)[:, None]
        ddy = self.oy[None, :] + 0.5 - (y - iy)[:, None]
        w = np.exp(-(ddx * ddx + ddy * ddy) / (2 * sig[:, None] ** 2)) * amp[:, None]
        px, py = ix[:, None] + self.ox[None, :], iy[:, None] + self.oy[None, :]
        ok = (px >= 0) & (px < W) & (py >= 0) & (py < H)
        flat = img.reshape(-1, 3)
        np.add.at(flat, (py * W + px)[ok], (w[ok][:, None] * hexrgb("#d8fbff")).astype(F32))

        span = H * 1.1
        for b in self.bubbles:
            ph = (u * b["k"] + b["ph"]) % 1.0
            y = H * 1.05 - ph * span
            x = b["X"] + b["wa"] * math.sin(b["wk"] * p + b["wp"])
            r = b["r"] * (0.8 + 0.4 * ph)
            R = int(r + 4)
            x0, y0 = int(x) - R, int(y) - R
            if y0 + 2 * R < 0 or y0 >= H or x0 + 2 * R < 0 or x0 >= W:
                continue
            gx = np.arange(x0, x0 + 2 * R + 1, dtype=F32) + 0.5 - x
            gy = np.arange(y0, y0 + 2 * R + 1, dtype=F32)[:, None] + 0.5 - y
            d = np.sqrt(gx * gx + gy * gy)
            ring = np.exp(-((d - r) / (1.1 * s + 0.4)) ** 2) * 0.55 + np.clip(1 - d / r, 0, 1) * 0.06
            hl = np.exp(-((gx + r * 0.35) ** 2 + (gy + r * 0.35) ** 2) / (2 * (r * 0.18 + 0.5) ** 2)) * 0.8
            val = (ring + hl) * (0.6 + 0.4 * (1 - y / H))
            ys0, xs0 = max(y0, 0), max(x0, 0)
            ys1, xs1 = min(y0 + 2 * R + 1, H), min(x0 + 2 * R + 1, W)
            img[ys0:ys1, xs0:xs1] += val[ys0 - y0:ys1 - y0, xs0 - x0:xs1 - x0, None] * hexrgb("#e6ffff")
        return img


SCENES = {c.key: c for c in (AuroraLake, NeonHorizon, MoonlitSea, LavaLamp, FireflyForest, SunlitDepths)}


# --------------------------------------------------------------------------
# rendering pipeline
# --------------------------------------------------------------------------
_SCENE: Scene | None = None


def _work(i):
    return _SCENE.frame_u8(i).tobytes()


def encode(scene: Scene, out: Path, crf: int, preset: str, jobs: int, frames: int | None):
    global _SCENE
    _SCENE = scene
    W, H, fps = scene.W, scene.H, scene.fps
    n = frames or scene.nframes
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-framerate", str(fps), "-i", "-",
        "-vf", "scale=out_color_matrix=bt709:out_range=tv:flags=accurate_rnd+full_chroma_int,format=yuv420p",
        "-c:v", "libx264", "-preset", preset, "-crf", str(crf), "-profile:v", "high", "-level:v", "5.1",
        "-x264-params", f"aq-mode=3:keyint={fps * 2}:min-keyint={fps * 2}:scenecut=0",
        "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709", "-color_range", "tv",
        "-metadata", f"title={scene.title}",
        "-metadata", "comment=Procedural seamless loop - Wallpaper-Bank scripts/generate-live-wallpapers.py",
        "-movflags", "+faststart", "-an", str(out),
    ]
    ff = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    t0 = time.time()
    ctx = mp.get_context("fork")
    with ctx.Pool(jobs) as pool:
        for i, buf in enumerate(pool.imap(_work, range(n), chunksize=1)):
            ff.stdin.write(buf)
            if i % 30 == 0 or i == n - 1:
                el = time.time() - t0
                print(f"\r  {scene.key}: frame {i + 1}/{n}  {el:6.0f}s  eta {el / (i + 1) * (n - i - 1):5.0f}s",
                      end="", flush=True)
    ff.stdin.close()
    if ff.wait() != 0:
        sys.exit(f"ffmpeg failed for {scene.key}")
    print(f"\n  wrote {out}  ({out.stat().st_size / 1e6:.1f} MB)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scenes", nargs="*", help="scene keys, or 'all'")
    ap.add_argument("--list", action="store_true", help="list scenes and exit")
    ap.add_argument("--size", help="WxH (default 3840x2160 or 2160x3840 by scene orientation)")
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--crf", type=int, default=20)
    ap.add_argument("--preset", default="slow")
    ap.add_argument("--jobs", type=int, default=max(1, (mp.cpu_count() or 2) - 1))
    ap.add_argument("--out", type=Path, help="output directory (default wallpapers/Live-Wallpapers/<Orientation>)")
    ap.add_argument("--preview", type=float, metavar="SCALE", help="write PNG stills at this scale instead of video")
    ap.add_argument("--frames", type=int, help="encode only the first N frames (testing)")
    a = ap.parse_args()

    if a.list or not a.scenes:
        for k, c in SCENES.items():
            print(f"  {k:16s} {c.orientation:10s} {c.blurb}")
        return
    keys = list(SCENES) if a.scenes == ["all"] else a.scenes
    for k in keys:
        if k not in SCENES:
            sys.exit(f"unknown scene '{k}' (try --list)")
        cls = SCENES[k]
        if a.size:
            W, H = (int(x) for x in a.size.lower().split("x"))
        else:
            W, H = (3840, 2160) if cls.orientation == "horizontal" else (2160, 3840)
        if a.preview:
            W, H = int(W * a.preview) // 2 * 2, int(H * a.preview) // 2 * 2
        t0 = time.time()
        scene = cls(W, H, a.seconds, a.fps)
        print(f"{cls.title}: {W}x{H}, {scene.nframes} frames, setup {time.time() - t0:.1f}s")
        orient = "Horizontal" if W >= H else "Vertical"
        if a.preview:
            out = a.out or Path(".")
            out.mkdir(parents=True, exist_ok=True)
            for q in (0.0, 0.37, 0.71):
                i = int(q * scene.nframes)
                t1 = time.time()
                Image.fromarray(scene.frame_u8(i)).save(out / f"{k}-{int(q * 100):02d}.png")
                print(f"  frame {i}: {time.time() - t1:.2f}s")
            continue
        out = (a.out or OUT_ROOT / orient) / f"{cls.title}-{orient}.mp4"
        encode(scene, out, a.crf, a.preset, a.jobs, a.frames)


if __name__ == "__main__":
    main()
