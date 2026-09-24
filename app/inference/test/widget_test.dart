// 前処理コアの単体テスト (docs/design/mobile_port_spec.md §3 の正解定義)。
// Phase 2 で PyTorch 参照とゴールデン一致させる際の足場。

import 'dart:async';
import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:image/image.dart' as img;
import 'package:raspicat_vla_app/src/action_chunk.dart';
import 'package:raspicat_vla_app/src/config.dart';
import 'package:raspicat_vla_app/src/grpc/edge_action_client.dart';
import 'package:raspicat_vla_app/src/preprocessing.dart';

void main() {
  test('normalizeChw は CHW・ImageNet 正規化で正しい長さ/値', () {
    final im = img.Image(width: 8, height: 8);
    img.fill(im, color: img.ColorRgb8(0, 0, 0)); // 全黒
    final out = normalizeChw(im, OmniVlaConfig.obsSize);
    const area = OmniVlaConfig.obsSize * OmniVlaConfig.obsSize;
    expect(out.length, 3 * area);
    // 黒 = (0 - mean)/std。R plane 先頭を確認。
    final expectedR =
        (0.0 - OmniVlaConfig.imagenetMean[0]) / OmniVlaConfig.imagenetStd[0];
    expect(out[0], closeTo(expectedR, 1e-5));
  });

  test('blackChw は normalizeChw(全黒) と一致', () {
    final black = blackChw(OmniVlaConfig.obsSize);
    final im = img.Image(width: 4, height: 4);
    img.fill(im, color: img.ColorRgb8(0, 0, 0));
    final viaNorm = normalizeChw(im, OmniVlaConfig.obsSize);
    expect(black[0], closeTo(viaNorm[0], 1e-6));
    expect(black.length, viaNorm.length);
  });

  test('poseGoalVector は (x_fwd/s, y_left/s, cos, sin) でクランプ', () {
    // x=前方2m, y=左1m, theta=0。goal_pose は waypoint と同じ (前方, 左) 系。
    final v = poseGoalVector([2.0, 1.0, 0.0]);
    expect(v[0], closeTo(2.0 / OmniVlaConfig.metricWaypointSpacing, 1e-4));
    expect(v[1], closeTo(1.0 / OmniVlaConfig.metricWaypointSpacing, 1e-4));
    expect(v[2], closeTo(1.0, 1e-6)); // cos0
    expect(v[3], closeTo(0.0, 1e-6)); // sin0

    // 40m 先 -> 30m にクランプ (方向 45deg)。
    final far = poseGoalVector([40.0, 40.0, 0.0]);
    final r = OmniVlaConfig.goalDistThresholdM / 1.41421356; // 各成分
    expect(far[0], closeTo(r / OmniVlaConfig.metricWaypointSpacing, 1e-2));
  });

  test('ObsRingBuffer は最大 historyLen で前詰め stack', () {
    final ring = ObsRingBuffer();
    const area = OmniVlaConfig.obsSize * OmniVlaConfig.obsSize;
    final frame = Float32List(3 * area)..fillRange(0, 3 * area, 0.5);
    ring.push(frame);
    final stacked = ring.stack();
    expect(stacked.length, 3 * OmniVlaConfig.historyLen * area);
    // 1 フレームしか無いので全ブロックが同じ値 (前詰め複製)。
    expect(stacked[0], closeTo(0.5, 1e-6));
    expect(stacked[stacked.length - 1], closeTo(0.5, 1e-6));
  });

  test('ActionChunk はメートル換算を返す', () {
    final raw = Float32List(
      OmniVlaConfig.lenTrajPred * OmniVlaConfig.actionDim,
    );
    raw[0] = 10.0; // x=10 units
    raw[1] = 0.0;
    final chunk = ActionChunk(raw);
    final xy = chunk.xyMetres.first;
    expect(xy.$1, closeTo(1.0, 1e-6)); // 10 * 0.1m
  });

  test('packFp16 は要素あたり 2 byte', () {
    final vals = Float32List.fromList([0.0, 1.0, -1.0, 0.5]);
    final bytes = packFp16(vals);
    expect(bytes.length, vals.length * 2);
  });

  test('CoalescingSender は close 後に保留 chunk を送らない', () async {
    final client = _RecordingClient();
    final sender = CoalescingSender(
      client,
      minInterval: const Duration(milliseconds: 50),
    );
    final chunk = ActionChunk(
      Float32List(OmniVlaConfig.lenTrajPred * OmniVlaConfig.actionDim),
    );
    sender.submit(chunk, frameId: 1, goalId: 'g');
    sender.submit(chunk, frameId: 2, goalId: 'g');
    await sender.close();
    sender.submit(chunk, frameId: 3, goalId: 'g');
    await Future<void>.delayed(const Duration(milliseconds: 80));
    expect(client.sent, [1]);
  });

  test('CoalescingSender は送信中の chunk 完了後に接続を閉じる', () async {
    final client = _RecordingClient()..pending = Completer<void>();
    final sender = CoalescingSender(client);
    final chunk = ActionChunk(
      Float32List(OmniVlaConfig.lenTrajPred * OmniVlaConfig.actionDim),
    );
    sender.submit(chunk, frameId: 1, goalId: 'g');
    final closing = sender.close();
    expect(client.closed, isFalse);
    client.pending!.complete();
    await closing;
    expect(client.closed, isTrue);
  });
}

class _RecordingClient implements EdgeActionClient {
  final List<int> sent = [];
  Completer<void>? pending;
  bool closed = false;

  @override
  Future<void> connect() async {}

  @override
  Future<void> send(
    ActionChunk chunk, {
    required int frameId,
    required String goalId,
  }) async {
    sent.add(frameId);
    await pending?.future;
  }

  @override
  String get status => 'test';

  @override
  Future<void> close() async {
    closed = true;
  }
}
