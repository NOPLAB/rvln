/**
 * ブラウザ -> Pi への action chunk 送信経路。
 * `app/lib/src/grpc/edge_action_client.dart` の移植。
 *
 * 重要 (CLAUDE.md / grpc_client.py の性質を踏襲): **coalesce + pace**。
 * 最新の chunk だけを保持し、一定レートでのみ送信する。遅い/切れたリンクが
 * 制御ループを詰まらせないようにするための不変条件。ここでも守る。
 */

import type { ActionChunk } from './actionChunk';

/** fp16 little-endian へパック (WS/proto の values_fp16 用)。 */
export function packFp16(values: Float32Array): Uint8Array {
  const bytes = new DataView(new ArrayBuffer(values.length * 2));
  for (let i = 0; i < values.length; i++) {
    bytes.setUint16(i * 2, floatToHalf(values[i]), true);
  }
  return new Uint8Array(bytes.buffer);
}

export interface SendMeta {
  frameId: number;
  goalId: string;
}

/** 送信先の抽象。実体は WebSocket / ログ / テストダブル。 */
export interface EdgeActionClient {
  connect(): Promise<void>;
  send(chunk: ActionChunk, meta: SendMeta): Promise<void>;
  /** 直近の接続/追従ステータス (UI 表示用)。 */
  readonly status: string;
  close(): Promise<void>;
}

/** ブラウザ内ログのみ。既定。Pi 接続前の動作確認に使う。 */
export class LoggingEdgeClient implements EdgeActionClient {
  private statusText = 'logging (no Pi)';
  private count = 0;

  async connect(): Promise<void> {}

  async send(chunk: ActionChunk, meta: SendMeta): Promise<void> {
    this.count += 1;
    this.statusText = `sent #${meta.frameId} (${chunk.fromModel ? 'model' : 'dummy'}) x${this.count}`;
  }

  get status(): string {
    return this.statusText;
  }

  async close(): Promise<void> {}
}

/**
 * 最新 chunk のみ保持し最大レートで送る coalescing/pacing ラッパー。
 * submit は即時 return (制御ループを塞がない)。送信中に来た新しい chunk は
 * 古いものを上書きする。
 */
export class CoalescingSender {
  private pending: { chunk: ActionChunk; meta: SendMeta } | null = null;
  private sending = false;
  private lastSent = 0;
  private closed = false;
  private draining: Promise<void> | null = null;
  private sendError: string | null = null;

  constructor(
    private readonly client: EdgeActionClient,
    private readonly minIntervalMs = 100,
  ) {}

  get status(): string {
    return this.sendError ?? this.client.status;
  }

  /** 送信キューへ投入 (最新のみ保持)。 */
  submit(chunk: ActionChunk, meta: SendMeta): void {
    if (this.closed) return;
    this.pending = { chunk, meta };
    this.startDrain();
  }

  private startDrain(): void {
    if (this.draining === null) {
      this.draining = this.drain().finally(() => {
        this.draining = null;
        if (!this.closed && this.pending !== null) this.startDrain();
      });
    }
  }

  clearPending(): void {
    this.pending = null;
  }

  private async drain(): Promise<void> {
    if (this.sending) return;
    this.sending = true;
    try {
      while (!this.closed && this.pending !== null) {
        const since = Date.now() - this.lastSent;
        if (since < this.minIntervalMs) {
          await sleep(this.minIntervalMs - since);
        }
        const item = this.pending;
        if (this.closed || item === null) break;
        this.pending = null;
        this.lastSent = Date.now();
        try {
          await this.client.send(item.chunk, item.meta);
          this.sendError = null;
        } catch (e) {
          this.sendError = `send error: ${e instanceof Error ? e.message : e}`;
        }
      }
    } finally {
      this.sending = false;
    }
  }

  async close(): Promise<void> {
    this.closed = true;
    this.clearPending();
    try {
      await this.draining;
    } finally {
      await this.client.close();
    }
  }
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

// --- IEEE754 float32 -> float16 (half) ---
const F32_VIEW = new DataView(new ArrayBuffer(4));

export function floatToHalf(value: number): number {
  F32_VIEW.setFloat32(0, value, true);
  const bits = F32_VIEW.getUint32(0, true);
  const sign = (bits >>> 16) & 0x8000;
  const exp = ((bits >>> 23) & 0xff) - 127 + 15;
  const mant = bits & 0x7fffff;
  if (exp <= 0) {
    // subnormal / zero にフラッシュ。
    return sign;
  }
  if (exp >= 0x1f) {
    // inf/nan。
    return sign | 0x7c00;
  }
  return sign | (exp << 10) | (mant >>> 13);
}
