'use client';

/**
 * メイン画面: 映像ソース + 推論ループ + パス可視化 + Pi 送信。
 * `app/lib/src/ui/home_page.dart` の移植。
 *
 * 観測は OBS_RATE_HZ (2Hz) で処理する。<video> の最新フレームだけを使い、
 * busy ガードで推論の多重実行を防ぐ (Flutter 版と同じ)。
 */

import { useCallback, useEffect, useRef, useState } from 'react';

import type { ActionChunk } from '@/lib/actionChunk';
import {
  CoalescingSender,
  type EdgeActionClient,
  LoggingEdgeClient,
} from '@/lib/edgeSender';
import { OmniVlaEngine, type OrtInitProgress } from '@/lib/engine';
import type { Goal } from '@/lib/goal';
import { clearModelCache } from '@/lib/modelStore';
import type { RgbaImage } from '@/lib/preprocessing';
import { WsEdgeClient } from '@/lib/wsEdgeClient';

import ControlPanel, { type VideoSource } from './ControlPanel';
import GoalPanel from './GoalPanel';
import PathOverlay from './PathOverlay';

/** 観測処理レート (Hz)。omnivla_edge_engine の obs_publish_rate に対応。 */
const OBS_RATE_HZ = 2.0;
/** 送信ペーシングの最小間隔 (ms)。Flutter 版 CoalescingSender と同値。 */
const SEND_MIN_INTERVAL_MS = 100;
const WS_URL_STORAGE_KEY = 'raspicat-vla-ws-url';

// エンジンはモジュールシングルトン: StrictMode の二重マウントや HMR で
// 590MB のモデルロードを繰り返さないため。
let engineSingleton: OmniVlaEngine | null = null;
let engineInitPromise: Promise<void> | null = null;
const progressListeners = new Set<(p: OrtInitProgress) => void>();

function ensureEngine(): { engine: OmniVlaEngine; ready: Promise<void> } {
  if (engineSingleton === null || engineInitPromise === null) {
    engineSingleton = new OmniVlaEngine();
    engineInitPromise = engineSingleton.init((p) => {
      for (const l of progressListeners) l(p);
    });
  }
  return { engine: engineSingleton, ready: engineInitPromise };
}

export default function VlaApp() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const captureCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const engineRef = useRef<OmniVlaEngine | null>(null);
  const clientRef = useRef<EdgeActionClient>(new LoggingEdgeClient());
  const senderRef = useRef<CoalescingSender>(
    new CoalescingSender(clientRef.current, SEND_MIN_INTERVAL_MS),
  );
  const busyRef = useRef(false);
  const frameIdRef = useRef(0);
  const generationRef = useRef(0);
  const connectionRef = useRef(0);
  const goalRef = useRef<Goal | null>(null);
  const runningRef = useRef(true);
  const currentFrameRef = useRef<RgbaImage | null>(null);

  const [engineReady, setEngineReady] = useState(false);
  const [engineStatus, setEngineStatus] = useState('初期化中…');
  const [initProgress, setInitProgress] = useState<OrtInitProgress | null>(
    null,
  );
  const [source, setSource] = useState<VideoSource>('camera');
  const [fileUrl, setFileUrl] = useState<string | null>(null);
  const [videoError, setVideoError] = useState('');
  const [goal, setGoal] = useState<Goal | null>(null);
  const [chunk, setChunk] = useState<ActionChunk | null>(null);
  const [latencyMs, setLatencyMs] = useState(0);
  const [tickError, setTickError] = useState('');
  const [running, setRunning] = useState(true);
  const [senderStatus, setSenderStatus] = useState('logging (no Pi)');
  const [wsUrl, setWsUrl] = useState('');
  const [wsConnected, setWsConnected] = useState(false);
  const [cacheNote, setCacheNote] = useState('');

  // --- エンジン初期化 (シングルトン) ---
  useEffect(() => {
    clientRef.current = new LoggingEdgeClient();
    senderRef.current = new CoalescingSender(
      clientRef.current,
      SEND_MIN_INTERVAL_MS,
    );
    const listener = (p: OrtInitProgress) => setInitProgress(p);
    progressListeners.add(listener);
    const { engine, ready } = ensureEngine();
    engineRef.current = engine;
    let cancelled = false;
    ready.then(() => {
      if (cancelled) return;
      setEngineReady(true);
      setInitProgress(null);
      setEngineStatus(
        engine.modelAvailable
          ? `ONNX [${engine.ep}] (text=${engine.textEncoderReady ? '有' : '無'})`
          : `ダミー (ONNX未配置${engine.lastError ? `: ${engine.lastError}` : ''})`,
      );
    });
    setWsUrl(localStorage.getItem(WS_URL_STORAGE_KEY) ?? '');
    return () => {
      cancelled = true;
      generationRef.current += 1;
      connectionRef.current += 1;
      void senderRef.current.close();
      progressListeners.delete(listener);
    };
  }, []);

  // --- 映像ソース: カメラ ---
  useEffect(() => {
    if (source !== 'camera') return;
    const video = videoRef.current;
    let stream: MediaStream | null = null;
    let cancelled = false;
    (async () => {
      try {
        stream = await navigator.mediaDevices.getUserMedia({
          video: {
            facingMode: 'environment',
            width: { ideal: 640 },
            height: { ideal: 480 },
          },
          audio: false,
        });
        if (cancelled) {
          for (const t of stream.getTracks()) t.stop();
          return;
        }
        if (video) {
          video.srcObject = stream;
          video.loop = false;
          await video.play().catch(() => {});
        }
        setVideoError('');
      } catch (e) {
        setVideoError(
          `カメラを開けません: ${e instanceof Error ? e.message : e} — 動画ファイルも使えます`,
        );
      }
    })();
    return () => {
      cancelled = true;
      for (const t of stream?.getTracks() ?? []) t.stop();
      if (video) video.srcObject = null;
    };
  }, [source]);

  // --- 映像ソース: 動画ファイル (ループ再生) ---
  useEffect(() => {
    if (source !== 'file' || fileUrl === null) return;
    const video = videoRef.current;
    if (!video) return;
    video.srcObject = null;
    video.src = fileUrl;
    video.loop = true;
    video.play().catch(() => {});
    setVideoError('');
    return () => {
      video.pause();
      video.removeAttribute('src');
    };
  }, [source, fileUrl]);

  const handleFile = useCallback(
    (file: File) => {
      if (fileUrl) URL.revokeObjectURL(fileUrl);
      setFileUrl(URL.createObjectURL(file));
      setSource('file');
    },
    [fileUrl],
  );

  // --- 推論 tick (2Hz) ---
  const tick = useCallback(async () => {
    const engine = engineRef.current;
    const video = videoRef.current;
    const goalNow = goalRef.current;
    const generation = generationRef.current;
    if (busyRef.current || !runningRef.current || !engine || !video || !goalNow)
      return;
    if (video.readyState < 2 || video.videoWidth === 0) return;
    busyRef.current = true;
    const t0 = performance.now();
    try {
      // 中央正方形クロップ (camera_image_utils.dart の centerCropToAspect(1.0) 相当)。
      const side = Math.min(video.videoWidth, video.videoHeight);
      let canvas = captureCanvasRef.current;
      if (!canvas) {
        canvas = document.createElement('canvas');
        captureCanvasRef.current = canvas;
      }
      if (canvas.width !== side || canvas.height !== side) {
        canvas.width = side;
        canvas.height = side;
      }
      const ctx = canvas.getContext('2d', { willReadFrequently: true });
      if (!ctx) throw new Error('2D コンテキストを取得できません');
      ctx.drawImage(
        video,
        (video.videoWidth - side) / 2,
        (video.videoHeight - side) / 2,
        side,
        side,
        0,
        0,
        side,
        side,
      );
      const frame = ctx.getImageData(0, 0, side, side);
      currentFrameRef.current = frame;

      const result = await engine.inferChunk(frame, goalNow);
      if (generation !== generationRef.current) return;
      frameIdRef.current += 1;
      senderRef.current.submit(result, {
        frameId: frameIdRef.current,
        goalId: goalNow.id,
      });

      setChunk(result);
      setLatencyMs(Math.round(performance.now() - t0));
      setSenderStatus(senderRef.current.status);
      setTickError('');
    } catch (e) {
      // 推論の一時エラーで画面全体を潰さない。
      setTickError(`推論エラー: ${e instanceof Error ? e.message : e}`);
    } finally {
      busyRef.current = false;
    }
  }, []);

  useEffect(() => {
    const id = setInterval(() => void tick(), Math.round(1000 / OBS_RATE_HZ));
    return () => clearInterval(id);
  }, [tick]);

  // --- ゴール設定 (履歴リセット) ---
  const handleGoal = useCallback((g: Goal) => {
    generationRef.current += 1;
    senderRef.current.clearPending();
    engineRef.current?.reset();
    goalRef.current = g;
    setGoal(g);
    setChunk(null);
  }, []);

  const handleToggleRunning = useCallback(() => {
    const next = !runningRef.current;
    if (!next) {
      generationRef.current += 1;
      senderRef.current.clearPending();
    }
    runningRef.current = next;
    setRunning(next);
  }, []);

  // --- Pi WebSocket 接続 ---
  const handleWsConnect = useCallback(() => {
    const url = wsUrl.trim();
    if (url === '') return;
    localStorage.setItem(WS_URL_STORAGE_KEY, url);
    generationRef.current += 1;
    const connection = ++connectionRef.current;
    const old = senderRef.current;
    void (async () => {
      await old.close();
      if (connection !== connectionRef.current) return;
      const client = new WsEdgeClient(url);
      clientRef.current = client;
      senderRef.current = new CoalescingSender(client, SEND_MIN_INTERVAL_MS);
      await client.connect();
      if (connection !== connectionRef.current) return;
      setWsConnected(true);
      setSenderStatus(client.status);
    })();
  }, [wsUrl]);

  const handleWsDisconnect = useCallback(() => {
    generationRef.current += 1;
    const connection = ++connectionRef.current;
    const old = senderRef.current;
    void (async () => {
      await old.close();
      if (connection !== connectionRef.current) return;
      const client = new LoggingEdgeClient();
      clientRef.current = client;
      senderRef.current = new CoalescingSender(client, SEND_MIN_INTERVAL_MS);
      setWsConnected(false);
      setSenderStatus(client.status);
    })();
  }, []);

  const handleClearCache = useCallback(() => {
    void clearModelCache().then((ok) => {
      setCacheNote(
        ok ? '削除しました (次回リロードで再取得)' : 'キャッシュはありません',
      );
    });
  }, []);

  // 総量は content-length か manifest.json 由来 (modelStore 参照)。どちらも
  // 無いときは取得済み MB のみ出す。丸め誤差で 100 を超えないようクランプ。
  const progressPct = initProgress?.fetch?.total
    ? Math.min(
        100,
        Math.round(
          (initProgress.fetch.loaded / initProgress.fetch.total) * 100,
        ),
      )
    : null;
  const progressMb = initProgress?.fetch
    ? Math.round(initProgress.fetch.loaded / 1048576)
    : null;

  return (
    <main className="app">
      <header className="app-header">
        <h1>Raspicat OmniVLA — Web Console</h1>
        <span
          className={`badge ${engineReady ? (chunk?.fromModel ? 'ok' : 'warn') : ''}`}
        >
          {engineStatus}
        </span>
        <p className="spec">
          omnivla-edge / action chunk 8×4 / 0.1 m per unit / onnxruntime-web
        </p>
      </header>

      <div className="content">
        <section>
          <div className="viewport">
            {/* muted/playsInline: 自動再生を許可させる */}
            <video ref={videoRef} muted playsInline />
            {chunk !== null && (
              <PathOverlay
                waypoints={chunk.xyMetres}
                fromModel={chunk.fromModel}
              />
            )}
            <div className="status-bar">
              <div>
                エンジン: {engineStatus} 推論: {latencyMs}ms
                {running ? '' : ' [停止中]'}
              </div>
              <div>ゴール: {goal?.id ?? '未設定'}</div>
              <div>送信: {senderStatus}</div>
              {initProgress && (
                <div>
                  {initProgress.stage}
                  {progressPct !== null
                    ? ` ${progressPct}% (${progressMb}MB)`
                    : progressMb !== null
                      ? ` ${progressMb}MB`
                      : ''}
                  {initProgress.fetch?.fromCache ? ' (キャッシュ)' : ''}
                  {progressPct !== null && (
                    <div className="progress">
                      <div style={{ width: `${progressPct}%` }} />
                    </div>
                  )}
                </div>
              )}
              {videoError && <div className="error">{videoError}</div>}
              {tickError && <div className="error">{tickError}</div>}
            </div>
            {goal === null && (
              <div className="viewport-placeholder">
                <p>
                  右のパネルからゴール (text / pose / image)
                  を設定すると推論が始まります。
                </p>
              </div>
            )}
          </div>
        </section>

        <aside>
          <ControlPanel
            source={source}
            onSource={setSource}
            onFile={handleFile}
            wsUrl={wsUrl}
            onWsUrl={setWsUrl}
            wsConnected={wsConnected}
            onWsConnect={handleWsConnect}
            onWsDisconnect={handleWsDisconnect}
            running={running}
            onToggleRunning={handleToggleRunning}
            onClearCache={handleClearCache}
            cacheNote={cacheNote}
          />
          <GoalPanel
            onGoal={handleGoal}
            getCurrentFrame={() => currentFrameRef.current}
          />
        </aside>
      </div>
    </main>
  );
}
