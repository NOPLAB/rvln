// This is a generated file - do not edit.
//
// Generated from edge_action.proto.

// @dart = 3.3

// ignore_for_file: annotate_overrides, camel_case_types, comment_references
// ignore_for_file: constant_identifier_names
// ignore_for_file: curly_braces_in_flow_control_structures
// ignore_for_file: deprecated_member_use_from_same_package, library_prefixes
// ignore_for_file: non_constant_identifier_names, prefer_relative_imports

import 'dart:async' as $async;
import 'dart:core' as $core;

import 'package:grpc/service_api.dart' as $grpc;
import 'package:protobuf/protobuf.dart' as $pb;

import 'edge_action.pb.dart' as $0;

export 'edge_action.pb.dart';

@$pb.GrpcServiceName('rvln.edge.EdgeActionService')
class EdgeActionServiceClient extends $grpc.Client {
  /// The hostname for this service.
  static const $core.String defaultHost = '';

  /// OAuth scopes needed for the client.
  static const $core.List<$core.String> oauthScopes = [
    '',
  ];

  EdgeActionServiceClient(super.channel, {super.options, super.interceptors});

  /// Phone streams action chunks to Pi; Pi streams control acknowledgements back.
  $grpc.ResponseStream<$0.ControlAck> streamActions(
    $async.Stream<$0.ActionChunk> request, {
    $grpc.CallOptions? options,
  }) {
    return $createStreamingCall(_$streamActions, request, options: options);
  }

  // method descriptors

  static final _$streamActions =
      $grpc.ClientMethod<$0.ActionChunk, $0.ControlAck>(
          '/rvln.edge.EdgeActionService/StreamActions',
          ($0.ActionChunk value) => value.writeToBuffer(),
          $0.ControlAck.fromBuffer);
}

@$pb.GrpcServiceName('rvln.edge.EdgeActionService')
abstract class EdgeActionServiceBase extends $grpc.Service {
  $core.String get $name => 'rvln.edge.EdgeActionService';

  EdgeActionServiceBase() {
    $addMethod($grpc.ServiceMethod<$0.ActionChunk, $0.ControlAck>(
        'StreamActions',
        streamActions,
        true,
        true,
        ($core.List<$core.int> value) => $0.ActionChunk.fromBuffer(value),
        ($0.ControlAck value) => value.writeToBuffer()));
  }

  $async.Stream<$0.ControlAck> streamActions(
      $grpc.ServiceCall call, $async.Stream<$0.ActionChunk> request);
}
