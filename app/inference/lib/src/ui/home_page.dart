/// メイン画面: カメラプレビュー + 推論ループ + パス可視化 + Pi 送信。
///
/// 観測は obs_publish_rate (~2Hz) で処理する。CameraImage 変換と推論を毎フレーム
/// 走らせない (負荷) ため、最新フレームだけを保持しタイマで tick する。
/// TODO(Phase 5): 変換/推論を Isolate に逃がして UI ジャンクを消す。
library;

import 'dart:async';
import 'dart:io' show Platform;

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:image/image.dart' as img;
import 'package:permission_handler/permission_handler.dart';

import '../action_chunk.dart';
import '../camera_image_utils.dart';
import '../goal.dart';
import '../grpc/edge_action_client.dart';
import '../omnivla_engine.dart';
import 'goal_panel.dart';
import 'path_painter.dart';

/// 観測処理レート (Hz)。omnivla_edge_engine の obs_publish_rate に対応。
const _obsRateHz = 2.0;

class HomePage extends StatefulWidget {
  const HomePage({super.key, required this.cameras});

  final List<CameraDescription> cameras;

  @override
  State<HomePage> createState() => _HomePageState();
}

class _HomePageState extends State<HomePage> with WidgetsBindingObserver {
  CameraController? _controller;
  final OmniVlaEngine _engine = OmniVlaEngine();
  CoalescingSender _sender = CoalescingSender(
    LoggingEdgeClient(),
    minInterval: const Duration(milliseconds: 100),
  );

  /// 接続中の Pi (EdgeActionService) のアドレス。空 = 未接続 (端末内ログのみ)。
  String _piAddress = '';

  Goal? _goal;
  CameraImage? _latestCameraImage;
  img.Image? _currentFrame; // image ゴール用
  ActionChunk? _chunk;
  Timer? _loop;
  bool _busy = false;
  int _frameId = 0;
  int _generation = 0;
  int _connectionGeneration = 0;
  int _lastLatencyMs = 0;
  String _error = '';
  String _tickError = '';
  bool _engineReady = false;
  bool _running = true; // 推論ループの再生/停止トグル
  String _engineStatus = '初期化中';

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    _init();
  }

  Future<void> _init() async {
    await _engine.init();
    if (!mounted) return;
    _engineReady = true;
    setState(() {
      _engineStatus = _engine.modelAvailable
          ? 'ONNX (text=${_engine.textEncoderReady ? "有" : "無"})'
          : 'ダミー (ONNX未配置)';
    });
    await _startCamera();
  }

  /// カメラのみ起動 (モデルは _init で 1 度だけロード済み)。resume 時にも呼ぶ。
  Future<void> _startCamera() async {
    if (widget.cameras.isEmpty) {
      setState(() => _error = 'カメラが見つかりません');
      return;
    }
    final granted = await Permission.camera.request();
    if (!mounted) return;
    if (!granted.isGranted) {
      setState(() => _error = 'カメラ権限がありません');
      return;
    }

    final back = widget.cameras.firstWhere(
      (c) => c.lensDirection == CameraLensDirection.back,
      orElse: () => widget.cameras.first,
    );
    // 一部端末 (CameraX) は Preview+ImageCapture+ImageAnalysis の 3 ストリームを
    // 高解像度で同時に張れず "too many use cases" になる。低解像度から試して
    // 段階的にフォールバックする。
    CameraController? controller;
    for (final preset in [ResolutionPreset.low, ResolutionPreset.medium]) {
      final c = CameraController(
        back,
        preset,
        enableAudio: false,
        // iOS の yuv420 は 2 プレーン NV12 (video-range) で来るため、ネイティブに
        // 安くフルレンジな BGRA を使う。Android は従来どおり YUV420。
        imageFormatGroup: Platform.isIOS
            ? ImageFormatGroup.bgra8888
            : ImageFormatGroup.yuv420,
      );
      try {
        await c.initialize();
        await c.startImageStream((image) => _latestCameraImage = image);
        controller = c;
        break;
      } catch (e) {
        await c.dispose();
        if (preset == ResolutionPreset.medium) {
          setState(() => _error = 'カメラ初期化失敗: $e');
          return;
        }
      }
    }
    if (!mounted) {
      await controller?.dispose();
      return;
    }
    if (controller == null) return;
    setState(() {
      _error = '';
      _controller = controller;
    });

    // タイマは 1 本だけ (resume での重複起動を防ぐ)。
    _loop ??= Timer.periodic(
      Duration(milliseconds: (1000 / _obsRateHz).round()),
      (_) => _tick(),
    );
  }

  Future<void> _tick() async {
    if (_busy || !_running) return;
    final goal = _goal;
    final generation = _generation;
    final camImage = _latestCameraImage;
    if (goal == null || camImage == null) return;
    _busy = true;
    final sw = Stopwatch()..start();
    try {
      final rgb = centerCropToAspect(cameraImageToRgb(camImage));
      _currentFrame = rgb;
      final chunk = await _engine.inferChunk(rgb, goal);
      if (!mounted || generation != _generation || !_running) return;
      _frameId++;
      _sender.submit(chunk, frameId: _frameId, goalId: goal.id);
      sw.stop();
      if (mounted) {
        setState(() {
          _chunk = chunk;
          _lastLatencyMs = sw.elapsedMilliseconds;
          _tickError = '';
        });
      }
    } catch (e) {
      // 推論の一時エラーで画面全体を潰さない (カメラは出したまま status に表示)。
      if (mounted) setState(() => _tickError = '推論エラー: $e');
    } finally {
      _busy = false;
    }
  }

  void _setGoal(Goal goal) {
    _generation++;
    _sender.clearPending();
    _engine.reset();
    setState(() {
      _goal = goal;
      _chunk = null;
    });
  }

  /// Pi (raspicat) の EdgeActionService へ gRPC 接続する。送信経路は
  /// CoalescingSender ごと差し替える (古い接続は close)。
  Future<void> _connectPi(String input) async {
    final trimmed = input.trim();
    if (trimmed.isEmpty) return;
    final sep = trimmed.lastIndexOf(':');
    final host = sep < 0 ? trimmed : trimmed.substring(0, sep);
    final port = sep < 0
        ? kDefaultEdgeActionPort
        : int.tryParse(trimmed.substring(sep + 1)) ?? kDefaultEdgeActionPort;

    final old = _sender;
    _generation++;
    _connectionGeneration++;
    final connectionGeneration = _connectionGeneration;
    await old.close();
    if (!mounted || connectionGeneration != _connectionGeneration) return;
    final client = GrpcEdgeClient(host, port: port);
    _sender = CoalescingSender(
      client,
      minInterval: const Duration(milliseconds: 100),
    );
    await client.connect();
    if (!mounted || connectionGeneration != _connectionGeneration) {
      await client.close();
      return;
    }
    setState(() => _piAddress = '$host:$port');
  }

  Future<void> _disconnectPi() async {
    final old = _sender;
    _generation++;
    _connectionGeneration++;
    final connectionGeneration = _connectionGeneration;
    await old.close();
    if (!mounted || connectionGeneration != _connectionGeneration) return;
    _sender = CoalescingSender(
      LoggingEdgeClient(),
      minInterval: const Duration(milliseconds: 100),
    );
    setState(() => _piAddress = '');
  }

  void _openPiDialog() {
    final controller = TextEditingController(
      text: _piAddress.isEmpty ? '' : _piAddress,
    );
    showDialog<void>(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('Raspberry Pi へ接続'),
        content: TextField(
          controller: controller,
          autofocus: true,
          keyboardType: TextInputType.url,
          decoration: const InputDecoration(
            labelText: 'IPアドレス[:ポート]',
            hintText: '例: 192.168.11.2:$kDefaultEdgeActionPort',
            helperText: 'vla.sh run omnivla_edge_mobile --mode cmd_vel の待受先',
          ),
          onSubmitted: (v) {
            Navigator.of(context).pop();
            _connectPi(v);
          },
        ),
        actions: [
          if (_piAddress.isNotEmpty)
            TextButton(
              onPressed: () {
                Navigator.of(context).pop();
                _disconnectPi();
              },
              child: const Text('切断'),
            ),
          TextButton(
            onPressed: () => Navigator.of(context).pop(),
            child: const Text('キャンセル'),
          ),
          FilledButton(
            onPressed: () {
              Navigator.of(context).pop();
              _connectPi(controller.text);
            },
            child: const Text('接続'),
          ),
        ],
      ),
    );
  }

  void _openGoalPanel() {
    showModalBottomSheet<void>(
      context: context,
      isScrollControlled: true,
      builder: (_) =>
          GoalPanel(onGoal: _setGoal, currentFrame: () => _currentFrame),
    );
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    if (state == AppLifecycleState.inactive ||
        state == AppLifecycleState.paused) {
      _generation++;
      _sender.clearPending();
      _latestCameraImage = null;
      // 画面 OFF 等ではカメラだけ解放 (モデルは保持)。
      _controller?.dispose();
      if (mounted) setState(() => _controller = null);
    } else if (state == AppLifecycleState.resumed) {
      // resume 時はカメラのみ再開。モデルの再ロードはしない (470MB, 高コスト)。
      if (_controller == null && _error.isEmpty && _engineReady) {
        _startCamera();
      }
    }
  }

  @override
  void dispose() {
    _generation++;
    _connectionGeneration++;
    WidgetsBinding.instance.removeObserver(this);
    _loop?.cancel();
    _controller?.dispose();
    _engine.dispose();
    _sender.close();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Raspicat OmniVLA (edge)'),
        actions: [
          IconButton(
            tooltip: _piAddress.isEmpty ? 'Pi へ接続' : 'Pi: $_piAddress',
            icon: Icon(
              _piAddress.isEmpty ? Icons.wifi_off : Icons.wifi,
              color: _piAddress.isEmpty ? null : Colors.lightGreenAccent,
            ),
            onPressed: _openPiDialog,
          ),
          IconButton(
            tooltip: _running ? '推論を停止' : '推論を再生',
            icon: Icon(_running ? Icons.pause_circle : Icons.play_circle),
            onPressed: () {
              if (_running) {
                _generation++;
                _sender.clearPending();
              }
              setState(() => _running = !_running);
            },
          ),
          IconButton(
            icon: const Icon(Icons.info_outline),
            onPressed: () => showAboutDialog(
              context: context,
              applicationName: 'Raspicat OmniVLA',
              children: const [
                Text('スマホがカメラ取得+OmniVLA-edge推論、Raspberry Piがモーター制御。'),
              ],
            ),
          ),
        ],
      ),
      floatingActionButton: FloatingActionButton.extended(
        onPressed: _openGoalPanel,
        icon: const Icon(Icons.flag),
        label: Text(_goal == null ? 'ゴール設定' : 'ゴール変更'),
      ),
      body: _buildBody(),
    );
  }

  Widget _buildBody() {
    if (_error.isNotEmpty) {
      return Center(
        child: Padding(
          padding: const EdgeInsets.all(24),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              const Icon(Icons.videocam_off, size: 48),
              const SizedBox(height: 12),
              Text(_error, textAlign: TextAlign.center),
              const SizedBox(height: 8),
              Text(
                '推論エンジン: $_engineStatus',
                style: Theme.of(context).textTheme.bodySmall,
              ),
            ],
          ),
        ),
      );
    }
    final controller = _controller;
    if (controller == null || !controller.value.isInitialized) {
      return const Center(child: CircularProgressIndicator());
    }
    return Stack(
      fit: StackFit.expand,
      children: [
        Center(
          child: AspectRatio(
            aspectRatio: controller.value.aspectRatio,
            child: Stack(
              fit: StackFit.expand,
              children: [
                CameraPreview(controller),
                if (_chunk != null)
                  CustomPaint(
                    painter: PathPainter(
                      waypoints: _chunk!.xyMetres,
                      fromModel: _chunk!.fromModel,
                    ),
                  ),
              ],
            ),
          ),
        ),
        Positioned(top: 8, left: 8, right: 8, child: _statusBar()),
      ],
    );
  }

  Widget _statusBar() {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
      decoration: BoxDecoration(
        color: Colors.black.withValues(alpha: 0.5),
        borderRadius: BorderRadius.circular(8),
      ),
      child: DefaultTextStyle(
        style: const TextStyle(fontSize: 12, color: Colors.white),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              'エンジン: $_engineStatus   推論: ${_lastLatencyMs}ms'
              '${_running ? "" : "   [停止中]"}',
            ),
            Text('ゴール: ${_goal?.id ?? "未設定"}'),
            Text('送信: ${_sender.status}'),
            if (_tickError.isNotEmpty)
              Text(
                _tickError,
                style: const TextStyle(color: Color(0xFFFF8A80)),
              ),
          ],
        ),
      ),
    );
  }
}
