/**
 * モデルバイト列の取得 + Cache Storage 永続化。
 *
 * omnivla_edge.onnx (345MB) + clip_text.onnx (243MB) を毎回ダウンロードしない
 * よう、初回取得時に Cache API へ保存し 2 回目以降はローカルから即ロードする。
 * Cache API が使えない文脈 (非 secure context 等) では素の fetch に落ちる。
 */

const CACHE_NAME = 'rvln-models-v1';

export interface FetchProgress {
  /** 取得済みバイト。 */
  loaded: number;
  /** Content-Length 不明なら null。 */
  total: number | null;
  /** true ならキャッシュヒット (ネットワーク非経由)。 */
  fromCache: boolean;
}

function cacheAvailable(): boolean {
  return typeof caches !== 'undefined';
}

/**
 * url を取得して Uint8Array を返す。404 やネットワーク失敗は null
 * (呼び出し側がダミー軌道へフォールバック)。
 *
 * expectedTotal は展開後の総バイト数 (manifest.json 由来)。応答ヘッダから
 * 総量が取れないときの進捗分母に使う。
 */
export async function fetchModelCached(
  url: string,
  onProgress?: (p: FetchProgress) => void,
  expectedTotal?: number,
): Promise<Uint8Array | null> {
  if (cacheAvailable()) {
    try {
      const cache = await caches.open(CACHE_NAME);
      const hit = await cache.match(url);
      if (hit) {
        const buf = await hit.arrayBuffer();
        onProgress?.({
          loaded: buf.byteLength,
          total: buf.byteLength,
          fromCache: true,
        });
        return new Uint8Array(buf);
      }
    } catch {
      // キャッシュ層の失敗は無視してネットワークへ。
    }
  }

  let resp: Response;
  try {
    resp = await fetch(url);
  } catch {
    return null;
  }
  if (!resp.ok || !resp.body) return null;

  // content-encoding (gzip 等) 付き応答では content-length は圧縮後サイズで、
  // reader が返す展開後バイト数と食い違い進捗が 100% を超える (GitHub Pages が
  // これに該当)。その場合は manifest 由来の展開後サイズを分母にする。
  const encoding = resp.headers.get('content-encoding');
  const headerTotal =
    !encoding || encoding === 'identity'
      ? Number(resp.headers.get('content-length')) || null
      : null;
  const total = headerTotal ?? expectedTotal ?? null;
  const reader = resp.body.getReader();
  const chunks: Uint8Array[] = [];
  let loaded = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    loaded += value.byteLength;
    onProgress?.({ loaded, total, fromCache: false });
  }
  const bytes = new Uint8Array(loaded);
  let offset = 0;
  for (const c of chunks) {
    bytes.set(c, offset);
    offset += c.byteLength;
  }

  if (cacheAvailable()) {
    try {
      const cache = await caches.open(CACHE_NAME);
      await cache.put(
        url,
        new Response(bytes, { headers: { 'content-length': `${loaded}` } }),
      );
    } catch {
      // 容量不足等。保存できなくても動作には影響しない。
    }
  }
  return bytes;
}

/** 保存済みモデルキャッシュを全消去する (~590MB)。 */
export async function clearModelCache(): Promise<boolean> {
  if (!cacheAvailable()) return false;
  return caches.delete(CACHE_NAME);
}
