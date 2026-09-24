import type { Metadata } from 'next';
import type { ReactNode } from 'react';

import './globals.css';

export const metadata: Metadata = {
  title: 'rvln — Web Console',
  description:
    'OmniVLA-edge をブラウザ内 (WebGPU / wasm) で推論し、action chunk を Raspberry Pi へ送る Web ポート。',
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="ja">
      <body>{children}</body>
    </html>
  );
}
