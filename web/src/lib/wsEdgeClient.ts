/**
 * WebSocket 実装の EdgeActionClient。Pi 側 `edge_action_ws_node` へ送る。
 *
 * ワイヤ形式は docs/design/web_port_spec.md §WS プロトコル (edge_action.proto の
 * ActionChunk/ControlAck と意味的に同一)。values は fp16 + base64 で、Pi 側は
 * 既存 `rvla_proto.conversions.fp16_bytes_to_float32_list` で復元する。
 *
 * 再接続は指数バックオフ (1s -> 5s 上限)。切断中の send は捨てる (coalesce+pace
 * は上位の CoalescingSender が担う; ここで溜めると詰まりの原因になる)。
 */

import type { ActionChunk } from './actionChunk';
import { type EdgeActionClient, packFp16, type SendMeta } from './edgeSender';

interface AckMessage {
  type: 'ack';
  frame_id: number;
  following: boolean;
  status: string;
}

export class WsEdgeClient implements EdgeActionClient {
  private ws: WebSocket | null = null;
  private closed = false;
  private retryMs = 1000;
  private statusText = 'connecting...';
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null;

  constructor(private readonly url: string) {}

  get status(): string {
    return `${this.url} ${this.statusText}`;
  }

  async connect(): Promise<void> {
    this.closed = false;
    this.open();
  }

  private open(): void {
    if (this.closed) return;
    let ws: WebSocket;
    try {
      ws = new WebSocket(this.url);
    } catch (e) {
      this.statusText = `invalid url (${e instanceof Error ? e.message : e})`;
      return;
    }
    this.ws = ws;
    ws.onopen = () => {
      this.retryMs = 1000;
      this.statusText = 'connected';
    };
    ws.onmessage = (ev) => {
      try {
        const msg = JSON.parse(String(ev.data)) as AckMessage;
        if (msg.type === 'ack') {
          this.statusText = `ack #${msg.frame_id} ${msg.status} following=${msg.following}`;
        }
      } catch {
        // ack 以外は無視。
      }
    };
    ws.onclose = () => {
      this.ws = null;
      if (this.closed) return;
      this.statusText = `disconnected (retry in ${Math.round(this.retryMs / 1000)}s)`;
      this.reconnectTimer = setTimeout(() => this.open(), this.retryMs);
      this.retryMs = Math.min(this.retryMs * 2, 5000);
    };
    ws.onerror = () => {
      // onclose が続くのでここでは状態だけ。
      this.statusText = 'error';
    };
  }

  async send(chunk: ActionChunk, meta: SendMeta): Promise<void> {
    const ws = this.ws;
    if (!ws || ws.readyState !== WebSocket.OPEN) return; // 切断中は捨てる
    ws.send(
      JSON.stringify({
        type: 'action_chunk',
        frame_id: meta.frameId,
        capture_time_ms: Date.now(),
        num_tokens: chunk.numTokens,
        embed_dim: chunk.embedDim,
        values_fp16_b64: bytesToBase64(packFp16(chunk.raw)),
        scaled_to_m: false, // 生 spacing 単位のまま送り Pi 側で ×0.1 する
        goal_id: meta.goalId,
      }),
    );
  }

  async close(): Promise<void> {
    this.closed = true;
    if (this.reconnectTimer !== null) clearTimeout(this.reconnectTimer);
    this.ws?.close();
    this.ws = null;
    this.statusText = 'closed';
  }
}

function bytesToBase64(bytes: Uint8Array): string {
  let bin = '';
  for (const b of bytes) bin += String.fromCharCode(b);
  return btoa(bin);
}
