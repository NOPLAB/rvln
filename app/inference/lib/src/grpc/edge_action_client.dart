/// スマホ -> Pi への action chunk 送信経路。
///
/// 重要 (CLAUDE.md / grpc_client.py の性質を踏襲): **coalesce + pace**。
/// 最新の chunk だけを保持し、一定レートでのみ送信する。遅い/切れたリンクが
/// 制御ループを詰まらせないようにするための不変条件。ここでも守る。
///
/// 実送信は [GrpcEdgeClient] (proto/edge_action.proto の
/// EdgeActionService.StreamActions、Pi 側は edge_action_grpc_node)。
/// Pi 未接続時は [LoggingEdgeClient] で端末内可視化のみ行い、アプリは動く。
library;

import 'dart:async';
import 'dart:typed_data';

import 'package:fixnum/fixnum.dart';
import 'package:grpc/grpc.dart';

import '../action_chunk.dart';
import 'gen/edge_action.pbgrpc.dart' as pb;

/// EdgeActionService の既定ポート (vla.sh の EDGE_ACTION_PORT と一致)。
const int kDefaultEdgeActionPort = 50061;

/// fp16 little-endian へパック (proto ActionChunk.values_fp16 用)。
Uint8List packFp16(Float32List values) {
  final bytes = ByteData(values.length * 2);
  for (var i = 0; i < values.length; i++) {
    bytes.setUint16(i * 2, _floatToHalf(values[i]), Endian.little);
  }
  return bytes.buffer.asUint8List();
}

/// 送信先の抽象。実体は gRPC / ログ / テストダブル。
abstract class EdgeActionClient {
  Future<void> connect();

  /// 1 chunk を送る。frameId / goalId はメタ。
  Future<void> send(
    ActionChunk chunk, {
    required int frameId,
    required String goalId,
  });

  /// 直近の接続/追従ステータス (UI 表示用)。
  String get status;

  Future<void> close();
}

/// 端末内ログのみ。既定。実機 Pi 接続前の動作確認に使う。
class LoggingEdgeClient implements EdgeActionClient {
  String _status = 'logging (no Pi)';
  int _count = 0;

  @override
  Future<void> connect() async {}

  @override
  Future<void> send(
    ActionChunk chunk, {
    required int frameId,
    required String goalId,
  }) async {
    _count++;
    _status =
        'sent #$frameId (${chunk.fromModel ? "model" : "dummy"}) x$_count';
  }

  @override
  String get status => _status;

  @override
  Future<void> close() async {}
}

/// Pi の EdgeActionService へ chunk を stream 送信する gRPC クライアント。
///
/// - x,y はスマホ側でメートル化して送る (`scaled_to_m=true`,
///   mobile_port_spec §5-D の推奨解)。Pi 側は両対応。
/// - 切断されても [send] は失敗させない: ストリームを破棄して次回 send で
///   張り直す (TCP 再接続は gRPC チャネルが担う)。制御ループ非停止の不変条件。
class GrpcEdgeClient implements EdgeActionClient {
  GrpcEdgeClient(this.host, {this.port = kDefaultEdgeActionPort});

  final String host;
  final int port;

  ClientChannel? _channel;
  pb.EdgeActionServiceClient? _stub;
  StreamController<pb.ActionChunk>? _requests;
  String _status = '未接続';

  @override
  Future<void> connect() async {
    _channel ??= ClientChannel(
      host,
      port: port,
      options: const ChannelOptions(
        credentials: ChannelCredentials.insecure(),
        connectionTimeout: Duration(seconds: 5),
        // Wi-Fi 断からの復帰を自動で拾う。
        backoffStrategy: defaultBackoffStrategy,
      ),
    );
    _stub ??= pb.EdgeActionServiceClient(_channel!);
    _status = '接続待ち $host:$port';
    _ensureStream();
  }

  /// 双方向ストリームを (再) 確立する。ack/エラーは status に反映するだけで、
  /// 送信側 (CoalescingSender) を止めない。
  void _ensureStream() {
    if (_requests != null) return;
    final requests = StreamController<pb.ActionChunk>();
    _requests = requests;
    final acks = _stub!.streamActions(requests.stream);
    acks.listen(
      (ack) {
        _status =
            'Pi応答 #${ack.frameId} '
            '${ack.following ? "追従中" : "停止"} (${ack.status})';
      },
      onError: (Object e) {
        _status = '切断: ${e is GrpcError ? (e.message ?? e.codeName) : e}';
        _dropStream(requests);
      },
      onDone: () {
        if (identical(_requests, requests)) _status = 'ストリーム終了';
        _dropStream(requests);
      },
      cancelOnError: true,
    );
  }

  void _dropStream(StreamController<pb.ActionChunk> requests) {
    if (identical(_requests, requests)) _requests = null;
    unawaited(requests.close());
  }

  @override
  Future<void> send(
    ActionChunk chunk, {
    required int frameId,
    required String goalId,
  }) async {
    if (_stub == null) await connect();
    _ensureStream();
    // メートル化した (x, y, cos, sin) × numTokens を fp16 でパック。
    final values = Float32List(chunk.numTokens * chunk.embedDim);
    for (var i = 0; i < chunk.numTokens; i++) {
      final row = chunk.rowMetres(i);
      for (var j = 0; j < chunk.embedDim; j++) {
        values[i * chunk.embedDim + j] = row[j];
      }
    }
    _requests!.add(
      pb.ActionChunk(
        frameId: Int64(frameId),
        captureTimeNs: Int64(DateTime.now().microsecondsSinceEpoch) * 1000,
        numTokens: chunk.numTokens,
        embedDim: chunk.embedDim,
        valuesFp16: packFp16(values),
        scaledToM: true,
        goalId: goalId,
        fromModel: chunk.fromModel,
      ),
    );
  }

  @override
  String get status => '$host:$port  $_status';

  @override
  Future<void> close() async {
    final requests = _requests;
    _requests = null;
    await requests?.close();
    await _channel?.shutdown();
    _channel = null;
    _stub = null;
    _status = '切断済み';
  }
}

/// 最新 chunk のみ保持し最大レートで送る coalescing/pacing ラッパー。
///
/// [submit] は即時 return (制御ループを塞がない)。実送信は内部タイマで行い、
/// 送信中に来た新しい chunk は古いものを上書きする。
class CoalescingSender {
  CoalescingSender(
    this._client, {
    this._minInterval = const Duration(milliseconds: 100),
  });

  final EdgeActionClient _client;
  final Duration _minInterval;

  ActionChunk? _pending;
  int _pendingFrameId = 0;
  String _pendingGoalId = '';
  bool _sending = false;
  bool _closed = false;
  Future<void>? _draining;
  String? _sendError;
  DateTime _lastSent = DateTime.fromMillisecondsSinceEpoch(0);

  String get status => _sendError ?? _client.status;

  /// 送信キューへ投入 (最新のみ保持)。
  void submit(
    ActionChunk chunk, {
    required int frameId,
    required String goalId,
  }) {
    if (_closed) return;
    _pending = chunk;
    _pendingFrameId = frameId;
    _pendingGoalId = goalId;
    _startDrain();
  }

  void _startDrain() {
    if (_draining == null) {
      final draining = _drain();
      _draining = draining;
      unawaited(
        draining.whenComplete(() {
          _draining = null;
          if (!_closed && _pending != null) _startDrain();
        }),
      );
    }
  }

  void clearPending() => _pending = null;

  Future<void> _drain() async {
    if (_sending) return;
    _sending = true;
    try {
      while (!_closed && _pending != null) {
        final now = DateTime.now();
        final since = now.difference(_lastSent);
        if (since < _minInterval) {
          await Future<void>.delayed(_minInterval - since);
        }
        final chunk = _pending;
        if (_closed || chunk == null) break;
        _pending = null;
        _lastSent = DateTime.now();
        try {
          await _client.send(
            chunk,
            frameId: _pendingFrameId,
            goalId: _pendingGoalId,
          );
          _sendError = null;
        } catch (e) {
          _sendError = 'send error: $e';
        }
      }
    } finally {
      _sending = false;
    }
  }

  Future<void> close() async {
    _closed = true;
    clearPending();
    try {
      await _draining;
    } finally {
      await _client.close();
    }
  }
}

// --- IEEE754 float32 -> float16 (half) ---
int _floatToHalf(double value) {
  final f = ByteData(4)..setFloat32(0, value, Endian.little);
  final bits = f.getUint32(0, Endian.little);
  final sign = (bits >> 16) & 0x8000;
  var exp = ((bits >> 23) & 0xff) - 127 + 15;
  var mant = bits & 0x7fffff;
  if (exp <= 0) {
    // subnormal / zero にフラッシュ。
    return sign;
  } else if (exp >= 0x1f) {
    // inf/nan。
    return sign | 0x7c00;
  }
  return sign | (exp << 10) | (mant >> 13);
}
