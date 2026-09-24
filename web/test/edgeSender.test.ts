import { describe, expect, it } from 'vitest';

import { ActionChunk } from '@/lib/actionChunk';
import { OmniVlaConfig } from '@/lib/config';
import {
  CoalescingSender,
  type EdgeActionClient,
  floatToHalf,
  LoggingEdgeClient,
  packFp16,
  type SendMeta,
} from '@/lib/edgeSender';

function chunkOf(fill: number): ActionChunk {
  return new ActionChunk(
    new Float32Array(OmniVlaConfig.lenTrajPred * OmniVlaConfig.actionDim).fill(
      fill,
    ),
    true,
  );
}

describe('floatToHalf / packFp16', () => {
  it('既知値の変換 (IEEE754 half)', () => {
    expect(floatToHalf(0)).toBe(0x0000);
    expect(floatToHalf(1.0)).toBe(0x3c00);
    expect(floatToHalf(-2.0)).toBe(0xc000);
    expect(floatToHalf(65504)).toBe(0x7bff); // half の最大値
    expect(floatToHalf(Infinity)).toBe(0x7c00);
    expect(floatToHalf(1e-8)).toBe(0x0000); // subnormal はゼロへフラッシュ
    expect(floatToHalf(-1e-8)).toBe(0x8000);
  });

  it('要素あたり 2 byte / little-endian', () => {
    const bytes = packFp16(Float32Array.of(1.0, -2.0));
    expect(bytes.length).toBe(4);
    expect(bytes[0]).toBe(0x00); // 0x3c00 LE
    expect(bytes[1]).toBe(0x3c);
    expect(bytes[2]).toBe(0x00); // 0xc000 LE
    expect(bytes[3]).toBe(0xc0);
  });
});

describe('CoalescingSender', () => {
  it('連投は最新のみ送られる (coalesce) / ペーシングが効く', async () => {
    const sent: Array<{ meta: SendMeta; v: number }> = [];
    const client: EdgeActionClient = {
      async connect() {},
      async send(chunk, meta) {
        sent.push({ meta, v: chunk.raw[0] });
      },
      get status() {
        return 'test';
      },
      async close() {},
    };
    const sender = new CoalescingSender(client, 30);

    sender.submit(chunkOf(1), { frameId: 1, goalId: 'g' });
    sender.submit(chunkOf(2), { frameId: 2, goalId: 'g' });
    sender.submit(chunkOf(3), { frameId: 3, goalId: 'g' });
    await new Promise((r) => setTimeout(r, 120));

    // 1 発目は即時、2,3 は coalesce されて 3 のみ。
    expect(sent.length).toBe(2);
    expect(sent[0].v).toBe(1);
    expect(sent[1].v).toBe(3);
    expect(sent[1].meta.frameId).toBe(3);
  });

  it('LoggingEdgeClient はステータスを更新する', async () => {
    const client = new LoggingEdgeClient();
    await client.send(chunkOf(1), { frameId: 7, goalId: 'g' });
    expect(client.status).toContain('#7');
  });

  it('close 後は保留 chunk と新規 chunk を送らない', async () => {
    const sent: number[] = [];
    const client: EdgeActionClient = {
      async connect() {},
      async send(_chunk, meta) {
        sent.push(meta.frameId);
      },
      get status() {
        return 'test';
      },
      async close() {},
    };
    const sender = new CoalescingSender(client, 50);
    sender.submit(chunkOf(1), { frameId: 1, goalId: 'g' });
    sender.submit(chunkOf(2), { frameId: 2, goalId: 'g' });
    await sender.close();
    sender.submit(chunkOf(3), { frameId: 3, goalId: 'g' });
    await new Promise((r) => setTimeout(r, 80));
    expect(sent).toEqual([1]);
  });

  it('close は送信中の chunk が終わるまで接続を閉じない', async () => {
    let finishSend: (() => void) | undefined;
    let closed = false;
    const client: EdgeActionClient = {
      async connect() {},
      async send() {
        await new Promise<void>((resolve) => {
          finishSend = resolve;
        });
      },
      get status() {
        return 'test';
      },
      async close() {
        closed = true;
      },
    };
    const sender = new CoalescingSender(client);
    sender.submit(chunkOf(1), { frameId: 1, goalId: 'g' });
    const closing = sender.close();
    expect(closed).toBe(false);
    finishSend?.();
    await closing;
    expect(closed).toBe(true);
  });
});
