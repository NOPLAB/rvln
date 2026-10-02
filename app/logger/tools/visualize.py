#!/usr/bin/env python3
"""vla_logger のセッションログをブラウザ GUI で可視化するツール。

`docs/design/logger_app_spec.md` §3 の on-disk 契約
(meta.json / camera/frames/*.jpg + frames.csv / pose/pose.csv /
imu/imu.csv / gnss/gnss.csv / audio/*.wav / labels.jsonl) をそのまま読み、
カメラ・VIO 姿勢(軌跡)・IMU・GNSS・音声・ラベルを共通の t_mono_ns 軸に
並べて再生する。

依存は Python 標準ライブラリのみ (matplotlib 等は不要)。ローカル HTTP
サーバをセッションディレクトリ直下に立て、埋め込みの単一 HTML ビューア
を配信する。file:// だと fetch/画像読み込みがブロックされるための構成。

使い方:
    python visualize.py <session_dir | session.zip> [--port 8000] [--no-open]

zip を渡した場合はテンポラリに展開し、内包する単一セッションディレクトリ
を自動で探す。Ctrl-C で終了 (展開先も後片付けする)。
"""

from __future__ import annotations

import argparse
import http.server
import shutil
import socketserver
import tempfile
import threading
import webbrowser
import zipfile
from pathlib import Path

# ---------------------------------------------------------------------------
# 埋め込みビューア (白黒モノトーンの銘板調。アクセント色は使わない)。
# ルート "/" でこの HTML を返し、それ以外は session ディレクトリのファイルを
# そのまま配信する。データ (meta.json 等) は相対パスで fetch する。
# ---------------------------------------------------------------------------
VIEWER_HTML = r"""<!doctype html>
<html lang="ja">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>vla_logger viewer</title>
<style>
  :root { --ink:#111; --line:#111; --mut:#666; --pane:#fafafa; }
  * { box-sizing: border-box; }
  html, body { margin:0; height:100%; background:#fff; color:var(--ink);
    font-family: system-ui, -apple-system, "Segoe UI", sans-serif; }
  .mono { font-family: ui-monospace, "SFMono-Regular", Menlo, Consolas, monospace; }
  #app { display:grid; grid-template-columns: minmax(320px, 1fr) 460px;
    grid-template-rows: auto 1fr auto; height:100%; gap:0; }
  header { grid-column: 1 / 3; display:flex; align-items:center; gap:16px;
    background:var(--ink); color:#fff; padding:8px 14px;
    text-transform:uppercase; letter-spacing:.14em; font-size:12px; }
  header .id { font-weight:700; letter-spacing:.2em; }
  header .stat { color:#ccc; letter-spacing:.08em; }
  header .stat b { color:#fff; font-weight:600; }
  #stage { grid-column:1; grid-row:2; background:#000; position:relative;
    display:flex; align-items:center; justify-content:center; overflow:hidden; }
  #frame { max-width:100%; max-height:100%; image-rendering:auto; }
  #stage .badge { position:absolute; left:10px; top:10px; background:#fff;
    color:#111; border:1px solid #111; padding:2px 8px; font-size:11px;
    letter-spacing:.1em; }
  #stage .label { position:absolute; left:10px; right:10px; bottom:10px;
    background:rgba(255,255,255,.94); color:#111; border:1px solid #111;
    padding:6px 10px; font-size:14px; min-height:0; }
  #stage .label:empty { display:none; }
  aside { grid-column:2; grid-row:2; border-left:1px solid var(--line);
    padding:12px; overflow:auto; display:flex; flex-direction:column; gap:12px; }
  .pane { border:1px solid var(--line); }
  .pane > h2 { margin:0; font-size:11px; letter-spacing:.16em;
    text-transform:uppercase; background:var(--ink); color:#fff;
    padding:4px 8px; }
  .pane > .body { padding:8px; }
  canvas { display:block; width:100%; }
  .kv { display:grid; grid-template-columns:auto 1fr; gap:2px 12px;
    font-size:12px; }
  .kv .k { color:var(--mut); }
  .kv .v { text-align:right; }
  footer { grid-column:1 / 3; grid-row:3; border-top:1px solid var(--line);
    padding:10px 14px; display:flex; align-items:center; gap:12px; }
  #scrub { flex:1; accent-color:#111; }
  #time { font-size:13px; min-width:150px; }
  button { font-family:inherit; font-size:13px; background:#fff; color:#111;
    border:1px solid #111; border-radius:3px; padding:6px 14px; cursor:pointer; }
  button:hover { background:#111; color:#fff; }
  button.on { background:#111; color:#fff; }
  select { font-family:inherit; border:1px solid #111; border-radius:3px;
    padding:5px 6px; background:#fff; }
  .track { position:relative; height:22px; border:1px solid var(--line);
    background:var(--pane); }
  .track .seg { position:absolute; top:0; bottom:0; }
  .track .seg.label { background:#111; opacity:.85; }
  .track .seg.audio { background:repeating-linear-gradient(45deg,#111,#111 3px,#fff 3px,#fff 6px);
    border:1px solid #111; }
  .track .cursor { position:absolute; top:-2px; bottom:-2px; width:1px;
    background:#111; }
  .track-row { display:grid; grid-template-columns:54px 1fr; gap:8px;
    align-items:center; font-size:11px; color:var(--mut); margin-top:6px; }
  #err { position:fixed; inset:0; background:#fff; color:#111; padding:24px;
    font-family:ui-monospace,monospace; white-space:pre-wrap; display:none; }
</style>
</head>
<body>
<div id="app">
  <header>
    <span class="id mono" id="hId">—</span>
    <span class="stat">DUR <b id="hDur" class="mono">—</b></span>
    <span class="stat">CAM <b id="hCam" class="mono">0</b></span>
    <span class="stat">IMU <b id="hImu" class="mono">0</b></span>
    <span class="stat">GNSS <b id="hGnss" class="mono">0</b></span>
    <span class="stat">POSE <b id="hPose" class="mono">0</b></span>
    <span class="stat">AUDIO <b id="hAud" class="mono">0</b></span>
    <span class="stat" style="margin-left:auto">vla_logger</span>
  </header>

  <div id="stage">
    <span class="badge mono" id="frameBadge">—</span>
    <img id="frame" alt="camera frame">
    <div class="label" id="labelOverlay"></div>
  </div>

  <aside>
    <div class="pane">
      <h2>IMU</h2>
      <div class="body">
        <canvas id="imuCanvas" height="150"></canvas>
        <div class="kv mono" style="margin-top:8px">
          <span class="k">accel (m/s²)</span><span class="v" id="vAcc">—</span>
          <span class="k">gyro (rad/s)</span><span class="v" id="vGyro">—</span>
          <span class="k">mag (µT)</span><span class="v" id="vMag">—</span>
        </div>
      </div>
    </div>

    <div class="pane">
      <h2>GNSS track</h2>
      <div class="body">
        <canvas id="gnssCanvas" height="200"></canvas>
        <div class="kv mono" style="margin-top:8px">
          <span class="k">lat, lon</span><span class="v" id="vLatLon">—</span>
          <span class="k">speed / bearing</span><span class="v" id="vSpeed">—</span>
          <span class="k">acc (m)</span><span class="v" id="vAccM">—</span>
        </div>
      </div>
    </div>

    <div class="pane">
      <h2>Trajectory (VIO)</h2>
      <div class="body">
        <canvas id="poseCanvas" height="220"></canvas>
        <div class="kv mono" style="margin-top:8px">
          <span class="k">x, z (m)</span><span class="v" id="vXZ">—</span>
          <span class="k">y up (m)</span><span class="v" id="vY">—</span>
          <span class="k">tracking</span><span class="v" id="vTrack">—</span>
        </div>
      </div>
    </div>

    <div class="pane">
      <h2>Timeline</h2>
      <div class="body">
        <div class="track" id="trackLabel"><div class="cursor" id="curL"></div></div>
        <div class="track-row"><span>labels</span><span id="lblLegend" class="mono"></span></div>
        <div class="track" id="trackAudio" style="margin-top:8px"><div class="cursor" id="curA"></div></div>
        <div class="track-row"><span>audio</span><span id="audLegend" class="mono"></span></div>
      </div>
    </div>
  </aside>

  <footer>
    <button id="play">▶ PLAY</button>
    <select id="speed" title="再生速度">
      <option value="0.5">0.5x</option>
      <option value="1" selected>1x</option>
      <option value="2">2x</option>
      <option value="4">4x</option>
    </select>
    <input type="range" id="scrub" min="0" max="1000" value="0">
    <span class="mono" id="time">0.000 / 0.000 s</span>
  </footer>
</div>
<div id="err"></div>

<script>
"use strict";
const $ = (id) => document.getElementById(id);
const NS = 1e9;

function fail(msg) {
  const e = $("err"); e.style.display = "block";
  e.textContent = "読み込みエラー\n\n" + msg;
}

async function fetchText(path) {
  const r = await fetch(path, { cache: "no-store" });
  if (!r.ok) throw new Error(path + " -> HTTP " + r.status);
  return r.text();
}
async function fetchJSON(path) { return JSON.parse(await fetchText(path)); }

// 単純な CSV パーサ (ヘッダ行 + 数値。文字列は素通し)。
function parseCSV(text) {
  const lines = text.trim().split(/\r?\n/);
  const head = lines[0].split(",");
  const rows = [];
  for (let i = 1; i < lines.length; i++) {
    if (!lines[i]) continue;
    const cols = lines[i].split(",");
    const o = {};
    for (let j = 0; j < head.length; j++) {
      const v = cols[j];
      const n = Number(v);
      o[head[j]] = (v === "" || Number.isNaN(n)) ? v : n;
    }
    rows.push(o);
  }
  return rows;
}

// t (秒) 以下で最大の index を二分探索 (times は秒昇順)。
function idxAtOrBefore(times, t) {
  let lo = 0, hi = times.length - 1, ans = 0;
  if (hi < 0) return -1;
  if (t < times[0]) return 0;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    if (times[mid] <= t) { ans = mid; lo = mid + 1; }
    else hi = mid - 1;
  }
  return ans;
}

const state = {
  dur: 0, t: 0, playing: false, speed: 1, lastRAF: 0,
  frames: [], frameTimes: [], imu: [], imuTimes: [],
  gnss: [], gnssTimes: [], pose: [], poseTimes: [], labels: [], audio: [],
};

function fmt(x, d = 3) { return (x == null || Number.isNaN(x)) ? "—" : x.toFixed(d); }

async function load() {
  const meta = await fetchJSON("meta.json");
  const endNs = meta.session_end_mono_ns || 0;

  const frames = parseCSV(await fetchText("camera/frames.csv"));
  state.frames = frames.map((r) => ({
    t: r.t_mono_ns / NS,
    src: "camera/frames/" + String(r.frame_no).padStart(8, "0") + ".jpg",
    no: r.frame_no,
  }));
  state.frameTimes = state.frames.map((f) => f.t);

  try {
    const imu = parseCSV(await fetchText("imu/imu.csv"));
    state.imu = imu; state.imuTimes = imu.map((r) => r.t_mono_ns / NS);
  } catch (e) { /* imu 無しでも動く */ }

  try {
    const g = parseCSV(await fetchText("gnss/gnss.csv"));
    state.gnss = g; state.gnssTimes = g.map((r) => r.t_mono_ns / NS);
  } catch (e) { /* gnss 無しでも動く */ }

  try {
    const p = parseCSV(await fetchText("pose/pose.csv"));
    state.pose = p; state.poseTimes = p.map((r) => r.t_mono_ns / NS);
  } catch (e) { /* pose 無しでも動く (旧セッション互換) */ }

  try {
    const txt = await fetchText("labels.jsonl");
    state.labels = txt.trim().split(/\r?\n/).filter(Boolean).map(JSON.parse);
  } catch (e) { /* labels 無しでも動く */ }
  state.audio = state.labels.filter((l) => l.audio_clip);

  const maxT = Math.max(
    endNs / NS,
    state.frameTimes.at(-1) || 0,
    state.imuTimes.at(-1) || 0,
    state.gnssTimes.at(-1) || 0,
    state.poseTimes.at(-1) || 0,
  );
  state.dur = maxT || 1;

  $("hId").textContent = meta.session_id || "(session)";
  $("hDur").textContent = fmt(state.dur, 1) + "s";
  $("hCam").textContent = state.frames.length;
  $("hImu").textContent = state.imu.length;
  $("hGnss").textContent = state.gnss.length;
  $("hPose").textContent = state.pose.length;
  $("hAud").textContent = state.audio.length;
  $("lblLegend").textContent = state.labels.length + " 件";
  $("audLegend").textContent = state.audio.length + " 区間";

  drawStaticTracks();
  seek(0);
  requestAnimationFrame(tick);
}

// ---- 描画 ---------------------------------------------------------------
function hidpiResize(cv, cssH) {
  const dpr = window.devicePixelRatio || 1;
  const w = cv.clientWidth || 400;
  cv.width = Math.round(w * dpr);
  cv.height = Math.round(cssH * dpr);
  const ctx = cv.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { ctx, w, h: cssH };
}

function drawImu(t) {
  const cv = $("imuCanvas");
  const { ctx, w, h } = hidpiResize(cv, 150);
  ctx.clearRect(0, 0, w, h);
  ctx.strokeStyle = "#ddd"; ctx.strokeRect(.5, .5, w - 1, h - 1);
  const rows = state.imu, times = state.imuTimes;
  if (!rows.length) { return; }
  // accel 3軸を濃淡グレーで。y は accel の min/max で自動スケール。
  const keys = ["ax", "ay", "az"], shades = ["#111", "#777", "#bbb"];
  let mn = Infinity, mx = -Infinity;
  for (const r of rows) for (const k of keys) {
    if (r[k] < mn) mn = r[k]; if (r[k] > mx) mx = r[k];
  }
  if (mn === mx) { mn -= 1; mx += 1; }
  const pad = (mx - mn) * 0.08; mn -= pad; mx += pad;
  const x = (tt) => (tt / state.dur) * (w - 2) + 1;
  const y = (v) => h - 2 - ((v - mn) / (mx - mn)) * (h - 4);
  // 0 線
  if (mn < 0 && mx > 0) {
    ctx.strokeStyle = "#eee"; ctx.beginPath();
    ctx.moveTo(1, y(0)); ctx.lineTo(w - 1, y(0)); ctx.stroke();
  }
  const step = Math.max(1, Math.floor(rows.length / (w * 2)));
  keys.forEach((k, ki) => {
    ctx.strokeStyle = shades[ki]; ctx.lineWidth = 1; ctx.beginPath();
    let first = true;
    for (let i = 0; i < rows.length; i += step) {
      const px = x(times[i]), py = y(rows[i][k]);
      if (first) { ctx.moveTo(px, py); first = false; } else ctx.lineTo(px, py);
    }
    ctx.stroke();
  });
  // 時刻カーソル
  ctx.strokeStyle = "#111"; ctx.lineWidth = 1; ctx.beginPath();
  ctx.moveTo(x(t), 1); ctx.lineTo(x(t), h - 1); ctx.stroke();
}

function drawGnss(t) {
  const cv = $("gnssCanvas");
  const { ctx, w, h } = hidpiResize(cv, 200);
  ctx.clearRect(0, 0, w, h);
  ctx.strokeStyle = "#ddd"; ctx.strokeRect(.5, .5, w - 1, h - 1);
  const g = state.gnss;
  if (!g.length) {
    ctx.fillStyle = "#999"; ctx.font = "12px ui-monospace, monospace";
    ctx.fillText("no GNSS fix", 10, 20); return;
  }
  let minLat = Infinity, maxLat = -Infinity, minLon = Infinity, maxLon = -Infinity;
  for (const r of g) {
    minLat = Math.min(minLat, r.lat); maxLat = Math.max(maxLat, r.lat);
    minLon = Math.min(minLon, r.lon); maxLon = Math.max(maxLon, r.lon);
  }
  const spanLat = Math.max(maxLat - minLat, 1e-6);
  const spanLon = Math.max(maxLon - minLon, 1e-6);
  const m = 16;
  // 緯度方向は上が北。経度=x, 緯度=y(反転)。アスペクトは無視して枠に収める。
  const X = (lon) => m + ((lon - minLon) / spanLon) * (w - 2 * m);
  const Y = (lat) => h - m - ((lat - minLat) / spanLat) * (h - 2 * m);
  // 経路
  ctx.strokeStyle = "#111"; ctx.lineWidth = 1.5; ctx.beginPath();
  g.forEach((r, i) => { const px = X(r.lon), py = Y(r.lat);
    if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py); });
  ctx.stroke();
  // 全点
  ctx.fillStyle = "#999";
  for (const r of g) { ctx.beginPath(); ctx.arc(X(r.lon), Y(r.lat), 2, 0, 7); ctx.fill(); }
  // 現在点 (時刻最近傍)
  const i = idxAtOrBefore(state.gnssTimes, t);
  if (i >= 0) {
    const r = g[i];
    ctx.fillStyle = "#111"; ctx.beginPath();
    ctx.arc(X(r.lon), Y(r.lat), 5, 0, 7); ctx.fill();
    ctx.strokeStyle = "#fff"; ctx.lineWidth = 1.5; ctx.stroke();
  }
}

function drawTrajectory(t) {
  const cv = $("poseCanvas");
  const { ctx, w, h } = hidpiResize(cv, 220);
  ctx.clearRect(0, 0, w, h);
  ctx.strokeStyle = "#ddd"; ctx.strokeRect(.5, .5, w - 1, h - 1);
  const p = state.pose;
  if (!p.length) {
    ctx.fillStyle = "#999"; ctx.font = "12px ui-monospace, monospace";
    ctx.fillText("no VIO pose", 10, 20); return;
  }
  // world 座標は ty が重力(上)。上面図は水平面 (tx, tz) を等尺で描く。
  let minX = Infinity, maxX = -Infinity, minZ = Infinity, maxZ = -Infinity;
  for (const r of p) {
    minX = Math.min(minX, r.tx); maxX = Math.max(maxX, r.tx);
    minZ = Math.min(minZ, r.tz); maxZ = Math.max(maxZ, r.tz);
  }
  const m = 16;
  const spanX = Math.max(maxX - minX, 1e-3);
  const spanZ = Math.max(maxZ - minZ, 1e-3);
  // アスペクト保持: x/z で共通スケール (軌跡形状を歪めない)。
  const s = Math.min((w - 2 * m) / spanX, (h - 2 * m) / spanZ);
  const cx = (minX + maxX) / 2, cz = (minZ + maxZ) / 2;
  const X = (x) => w / 2 + (x - cx) * s;
  // z は上向きを画面上方に (画面 y は下向きなので反転)。
  const Y = (z) => h / 2 - (z - cz) * s;
  // 経路
  ctx.strokeStyle = "#111"; ctx.lineWidth = 1.5; ctx.beginPath();
  p.forEach((r, i) => { const px = X(r.tx), py = Y(r.tz);
    if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py); });
  ctx.stroke();
  // 始点 (□) と終点方向の目印
  ctx.fillStyle = "#999";
  ctx.fillRect(X(p[0].tx) - 3, Y(p[0].tz) - 3, 6, 6);
  // 現在位置 (時刻最近傍)
  const i = idxAtOrBefore(state.poseTimes, t);
  if (i >= 0) {
    const r = p[i];
    ctx.fillStyle = "#111"; ctx.beginPath();
    ctx.arc(X(r.tx), Y(r.tz), 5, 0, 7); ctx.fill();
    ctx.strokeStyle = "#fff"; ctx.lineWidth = 1.5; ctx.stroke();
  }
  // スケールバー (1 m)
  ctx.strokeStyle = "#111"; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(m, h - 8); ctx.lineTo(m + s, h - 8); ctx.stroke();
  ctx.fillStyle = "#666"; ctx.font = "10px ui-monospace, monospace";
  ctx.fillText("1 m", m + s + 4, h - 5);
}

function drawStaticTracks() {
  const trackL = $("trackLabel"), trackA = $("trackAudio");
  // 既存 seg を消す (cursor は残す)
  trackL.querySelectorAll(".seg").forEach((e) => e.remove());
  trackA.querySelectorAll(".seg").forEach((e) => e.remove());
  const put = (track, cls, t0, t1, title) => {
    const s = document.createElement("div");
    s.className = "seg " + cls;
    const a = Math.max(0, Math.min(1, t0 / state.dur));
    const b = Math.max(0, Math.min(1, t1 / state.dur));
    s.style.left = (a * 100) + "%";
    s.style.width = Math.max(0.4, (b - a) * 100) + "%";
    s.title = title;
    track.appendChild(s);
  };
  for (const l of state.labels) {
    put(trackL, "label", (l.t_start_mono_ns || 0) / NS,
        (l.t_end_mono_ns || l.t_start_mono_ns || 0) / NS, l.prompt || "(空)");
  }
  for (const a of state.audio) {
    put(trackA, "audio", (a.t_start_mono_ns || 0) / NS,
        (a.t_end_mono_ns || 0) / NS, a.audio_clip);
  }
}

// ---- 現在時刻の反映 ------------------------------------------------------
let curSrc = "";
function seek(t) {
  state.t = Math.max(0, Math.min(state.dur, t));
  const t_ = state.t;

  // カメラ
  const fi = idxAtOrBefore(state.frameTimes, t_);
  if (fi >= 0) {
    const f = state.frames[fi];
    if (f.src !== curSrc) { $("frame").src = f.src; curSrc = f.src; }
    $("frameBadge").textContent =
      "#" + f.no + "  " + fmt(f.t, 2) + "s";
  }

  // IMU 数値
  const ii = idxAtOrBefore(state.imuTimes, t_);
  if (ii >= 0 && state.imu.length) {
    const r = state.imu[ii];
    $("vAcc").textContent = [r.ax, r.ay, r.az].map((v) => fmt(v, 2)).join(", ");
    $("vGyro").textContent = [r.gx, r.gy, r.gz].map((v) => fmt(v, 3)).join(", ");
    $("vMag").textContent = [r.mx, r.my, r.mz].map((v) => fmt(v, 1)).join(", ");
  }

  // GNSS 数値
  const gi = idxAtOrBefore(state.gnssTimes, t_);
  if (gi >= 0 && state.gnss.length) {
    const r = state.gnss[gi];
    $("vLatLon").textContent = fmt(r.lat, 6) + ", " + fmt(r.lon, 6);
    $("vSpeed").textContent = fmt(r.speed, 2) + " m/s  /  " + fmt(r.bearing, 0) + "°";
    $("vAccM").textContent = fmt(r.acc, 1);
  }

  // 姿勢 (VIO) 数値
  const pi = idxAtOrBefore(state.poseTimes, t_);
  if (pi >= 0 && state.pose.length) {
    const r = state.pose[pi];
    $("vXZ").textContent = fmt(r.tx, 2) + ", " + fmt(r.tz, 2);
    $("vY").textContent = fmt(r.ty, 2);
    $("vTrack").textContent = r.tracking_state || "—";
  }

  // ラベル (現在時刻を含む区間)
  const active = state.labels.find((l) =>
    t_ >= (l.t_start_mono_ns || 0) / NS && t_ <= (l.t_end_mono_ns || 0) / NS);
  const ov = $("labelOverlay");
  ov.textContent = active ? (active.prompt || "(プロンプト空)") +
    (active.audio_clip ? "  🎙" + active.audio_clip : "") : "";

  // カーソル / スライダ / 時刻表示
  const frac = state.t / state.dur;
  $("curL").style.left = (frac * 100) + "%";
  $("curA").style.left = (frac * 100) + "%";
  $("scrub").value = String(Math.round(frac * 1000));
  $("time").textContent = fmt(state.t, 3) + " / " + fmt(state.dur, 3) + " s";

  drawImu(t_);
  drawGnss(t_);
  drawTrajectory(t_);
}

function tick(now) {
  if (state.playing) {
    if (!state.lastRAF) state.lastRAF = now;
    const dt = (now - state.lastRAF) / 1000 * state.speed;
    state.lastRAF = now;
    let nt = state.t + dt;
    if (nt >= state.dur) { nt = state.dur; setPlaying(false); }
    seek(nt);
  } else {
    state.lastRAF = 0;
  }
  requestAnimationFrame(tick);
}

function setPlaying(p) {
  state.playing = p; state.lastRAF = 0;
  const b = $("play");
  b.textContent = p ? "⏸ PAUSE" : "▶ PLAY";
  b.classList.toggle("on", p);
}

// ---- 入力 ---------------------------------------------------------------
$("play").addEventListener("click", () => {
  if (!state.playing && state.t >= state.dur) seek(0);
  setPlaying(!state.playing);
});
$("speed").addEventListener("change", (e) => { state.speed = Number(e.target.value); });
$("scrub").addEventListener("input", (e) => {
  setPlaying(false);
  seek((Number(e.target.value) / 1000) * state.dur);
});
window.addEventListener("keydown", (e) => {
  if (e.code === "Space") { e.preventDefault(); $("play").click(); }
  else if (e.code === "ArrowRight") { setPlaying(false); seek(state.t + (e.shiftKey ? 1 : 0.1)); }
  else if (e.code === "ArrowLeft") { setPlaying(false); seek(state.t - (e.shiftKey ? 1 : 0.1)); }
});
window.addEventListener("resize", () => seek(state.t));

load().catch((e) => fail(String(e && e.stack || e)));
</script>
</body>
</html>
"""


def find_session_dir(root: Path) -> Path:
    """meta.json を含むディレクトリを root から探して返す。"""
    if (root / "meta.json").is_file():
        return root
    # 直下 1 階層だけ見る (zip 展開で top ディレクトリが 1 個のケース)。
    candidates = [p for p in root.iterdir() if p.is_dir() and (p / "meta.json").is_file()]
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise SystemExit(f"meta.json が見つかりません: {root}")
    raise SystemExit(
        "複数のセッションが見つかりました。1 つを直接指定してください:\n  "
        + "\n  ".join(str(c) for c in candidates)
    )


def make_handler(session_dir: Path):
    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *a, **kw):
            super().__init__(*a, directory=str(session_dir), **kw)

        def do_GET(self):  # noqa: N802 (http.server 命名規約)
            if self.path in ("/", "/index.html"):
                body = VIEWER_HTML.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            super().do_GET()

        def log_message(self, *a):  # サーバログを抑制
            pass

    return Handler


def main() -> None:
    ap = argparse.ArgumentParser(description="vla_logger セッションログの GUI ビューア")
    ap.add_argument("path", help="セッションディレクトリ または .zip")
    ap.add_argument("--port", type=int, default=8000, help="待受ポート (既定 8000)")
    ap.add_argument("--no-open", action="store_true", help="ブラウザを自動で開かない")
    args = ap.parse_args()

    src = Path(args.path).expanduser().resolve()
    if not src.exists():
        raise SystemExit(f"存在しません: {src}")

    tmp: str | None = None
    if src.is_file() and src.suffix.lower() == ".zip":
        tmp = tempfile.mkdtemp(prefix="vla_logger_view_")
        with zipfile.ZipFile(src) as zf:
            zf.extractall(tmp)
        session_dir = find_session_dir(Path(tmp))
    else:
        session_dir = find_session_dir(src)

    # ポートが埋まっていたら次を試す。
    port = args.port
    httpd = None
    for p in range(args.port, args.port + 20):
        try:
            httpd = socketserver.ThreadingTCPServer(("127.0.0.1", p), make_handler(session_dir))
            port = p
            break
        except OSError:
            continue
    if httpd is None:
        raise SystemExit(f"空きポートが見つかりません ({args.port}..{args.port + 19})")

    url = f"http://127.0.0.1:{port}/"
    print(f"session : {session_dir}")
    print(f"viewer  : {url}")
    print("Ctrl-C で終了")

    if not args.no_open:
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n終了")
    finally:
        httpd.server_close()
        if tmp:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
