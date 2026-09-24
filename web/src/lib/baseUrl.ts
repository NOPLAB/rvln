/**
 * サブパス配信 (GitHub Pages の https://<user>.github.io/<repo>/ 等) 対応。
 *
 * ビルド時に NEXT_PUBLIC_BASE_PATH (例: "/rvln") を渡すと、
 * next.config.mjs の basePath と、実行時に fetch する public/ 資産
 * (ORT ランタイム・モデル・CLIP 語彙) の URL が揃って前置される。
 * 未設定 (ローカル localhost 配信) では空文字。
 */

export const BASE_PATH = process.env.NEXT_PUBLIC_BASE_PATH ?? '';

export function withBase(path: string): string {
  return `${BASE_PATH}${path}`;
}
