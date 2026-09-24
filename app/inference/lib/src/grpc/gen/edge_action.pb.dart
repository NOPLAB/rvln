// This is a generated file - do not edit.
//
// Generated from edge_action.proto.

// @dart = 3.3

// ignore_for_file: annotate_overrides, camel_case_types, comment_references
// ignore_for_file: constant_identifier_names
// ignore_for_file: curly_braces_in_flow_control_structures
// ignore_for_file: deprecated_member_use_from_same_package, library_prefixes
// ignore_for_file: non_constant_identifier_names, prefer_relative_imports

import 'dart:core' as $core;

import 'package:fixnum/fixnum.dart' as $fixnum;
import 'package:protobuf/protobuf.dart' as $pb;

export 'package:protobuf/protobuf.dart' show GeneratedMessageGenericExtensions;

class ActionChunk extends $pb.GeneratedMessage {
  factory ActionChunk({
    $fixnum.Int64? frameId,
    $fixnum.Int64? captureTimeNs,
    $core.int? numTokens,
    $core.int? embedDim,
    $core.List<$core.int>? valuesFp16,
    $core.bool? scaledToM,
    $core.String? goalId,
    $core.bool? fromModel,
  }) {
    final result = create();
    if (frameId != null) result.frameId = frameId;
    if (captureTimeNs != null) result.captureTimeNs = captureTimeNs;
    if (numTokens != null) result.numTokens = numTokens;
    if (embedDim != null) result.embedDim = embedDim;
    if (valuesFp16 != null) result.valuesFp16 = valuesFp16;
    if (scaledToM != null) result.scaledToM = scaledToM;
    if (goalId != null) result.goalId = goalId;
    if (fromModel != null) result.fromModel = fromModel;
    return result;
  }

  ActionChunk._();

  factory ActionChunk.fromBuffer($core.List<$core.int> data,
          [$pb.ExtensionRegistry registry = $pb.ExtensionRegistry.EMPTY]) =>
      create()..mergeFromBuffer(data, registry);
  factory ActionChunk.fromJson($core.String json,
          [$pb.ExtensionRegistry registry = $pb.ExtensionRegistry.EMPTY]) =>
      create()..mergeFromJson(json, registry);

  static final $pb.BuilderInfo _i = $pb.BuilderInfo(
      _omitMessageNames ? '' : 'ActionChunk',
      package:
          const $pb.PackageName(_omitMessageNames ? '' : 'rvln.edge'),
      createEmptyInstance: create)
    ..a<$fixnum.Int64>(1, _omitFieldNames ? '' : 'frameId', $pb.PbFieldType.OU6,
        defaultOrMaker: $fixnum.Int64.ZERO)
    ..a<$fixnum.Int64>(
        2, _omitFieldNames ? '' : 'captureTimeNs', $pb.PbFieldType.OU6,
        defaultOrMaker: $fixnum.Int64.ZERO)
    ..aI(3, _omitFieldNames ? '' : 'numTokens', fieldType: $pb.PbFieldType.OU3)
    ..aI(4, _omitFieldNames ? '' : 'embedDim', fieldType: $pb.PbFieldType.OU3)
    ..a<$core.List<$core.int>>(
        5, _omitFieldNames ? '' : 'valuesFp16', $pb.PbFieldType.OY)
    ..aOB(6, _omitFieldNames ? '' : 'scaledToM')
    ..aOS(7, _omitFieldNames ? '' : 'goalId')
    ..aOB(8, _omitFieldNames ? '' : 'fromModel')
    ..hasRequiredFields = false;

  @$core.Deprecated('See https://github.com/google/protobuf.dart/issues/998.')
  ActionChunk clone() => deepCopy();
  @$core.Deprecated('See https://github.com/google/protobuf.dart/issues/998.')
  ActionChunk copyWith(void Function(ActionChunk) updates) =>
      super.copyWith((message) => updates(message as ActionChunk))
          as ActionChunk;

  @$core.override
  $pb.BuilderInfo get info_ => _i;

  @$core.pragma('dart2js:noInline')
  static ActionChunk create() => ActionChunk._();
  @$core.override
  ActionChunk createEmptyInstance() => create();
  @$core.pragma('dart2js:noInline')
  static ActionChunk getDefault() => _defaultInstance ??=
      $pb.GeneratedMessage.$_defaultFor<ActionChunk>(create);
  static ActionChunk? _defaultInstance;

  @$pb.TagNumber(1)
  $fixnum.Int64 get frameId => $_getI64(0);
  @$pb.TagNumber(1)
  set frameId($fixnum.Int64 value) => $_setInt64(0, value);
  @$pb.TagNumber(1)
  $core.bool hasFrameId() => $_has(0);
  @$pb.TagNumber(1)
  void clearFrameId() => $_clearField(1);

  @$pb.TagNumber(2)
  $fixnum.Int64 get captureTimeNs => $_getI64(1);
  @$pb.TagNumber(2)
  set captureTimeNs($fixnum.Int64 value) => $_setInt64(1, value);
  @$pb.TagNumber(2)
  $core.bool hasCaptureTimeNs() => $_has(1);
  @$pb.TagNumber(2)
  void clearCaptureTimeNs() => $_clearField(2);

  @$pb.TagNumber(3)
  $core.int get numTokens => $_getIZ(2);
  @$pb.TagNumber(3)
  set numTokens($core.int value) => $_setUnsignedInt32(2, value);
  @$pb.TagNumber(3)
  $core.bool hasNumTokens() => $_has(2);
  @$pb.TagNumber(3)
  void clearNumTokens() => $_clearField(3);

  @$pb.TagNumber(4)
  $core.int get embedDim => $_getIZ(3);
  @$pb.TagNumber(4)
  set embedDim($core.int value) => $_setUnsignedInt32(3, value);
  @$pb.TagNumber(4)
  $core.bool hasEmbedDim() => $_has(3);
  @$pb.TagNumber(4)
  void clearEmbedDim() => $_clearField(4);

  @$pb.TagNumber(5)
  $core.List<$core.int> get valuesFp16 => $_getN(4);
  @$pb.TagNumber(5)
  set valuesFp16($core.List<$core.int> value) => $_setBytes(4, value);
  @$pb.TagNumber(5)
  $core.bool hasValuesFp16() => $_has(4);
  @$pb.TagNumber(5)
  void clearValuesFp16() => $_clearField(5);

  @$pb.TagNumber(6)
  $core.bool get scaledToM => $_getBF(5);
  @$pb.TagNumber(6)
  set scaledToM($core.bool value) => $_setBool(5, value);
  @$pb.TagNumber(6)
  $core.bool hasScaledToM() => $_has(5);
  @$pb.TagNumber(6)
  void clearScaledToM() => $_clearField(6);

  @$pb.TagNumber(7)
  $core.String get goalId => $_getSZ(6);
  @$pb.TagNumber(7)
  set goalId($core.String value) => $_setString(6, value);
  @$pb.TagNumber(7)
  $core.bool hasGoalId() => $_has(6);
  @$pb.TagNumber(7)
  void clearGoalId() => $_clearField(7);

  @$pb.TagNumber(8)
  $core.bool get fromModel => $_getBF(7);
  @$pb.TagNumber(8)
  set fromModel($core.bool value) => $_setBool(7, value);
  @$pb.TagNumber(8)
  $core.bool hasFromModel() => $_has(7);
  @$pb.TagNumber(8)
  void clearFromModel() => $_clearField(8);
}

class ControlAck extends $pb.GeneratedMessage {
  factory ControlAck({
    $fixnum.Int64? frameId,
    $core.bool? following,
    $core.String? status,
  }) {
    final result = create();
    if (frameId != null) result.frameId = frameId;
    if (following != null) result.following = following;
    if (status != null) result.status = status;
    return result;
  }

  ControlAck._();

  factory ControlAck.fromBuffer($core.List<$core.int> data,
          [$pb.ExtensionRegistry registry = $pb.ExtensionRegistry.EMPTY]) =>
      create()..mergeFromBuffer(data, registry);
  factory ControlAck.fromJson($core.String json,
          [$pb.ExtensionRegistry registry = $pb.ExtensionRegistry.EMPTY]) =>
      create()..mergeFromJson(json, registry);

  static final $pb.BuilderInfo _i = $pb.BuilderInfo(
      _omitMessageNames ? '' : 'ControlAck',
      package:
          const $pb.PackageName(_omitMessageNames ? '' : 'rvln.edge'),
      createEmptyInstance: create)
    ..a<$fixnum.Int64>(1, _omitFieldNames ? '' : 'frameId', $pb.PbFieldType.OU6,
        defaultOrMaker: $fixnum.Int64.ZERO)
    ..aOB(2, _omitFieldNames ? '' : 'following')
    ..aOS(3, _omitFieldNames ? '' : 'status')
    ..hasRequiredFields = false;

  @$core.Deprecated('See https://github.com/google/protobuf.dart/issues/998.')
  ControlAck clone() => deepCopy();
  @$core.Deprecated('See https://github.com/google/protobuf.dart/issues/998.')
  ControlAck copyWith(void Function(ControlAck) updates) =>
      super.copyWith((message) => updates(message as ControlAck)) as ControlAck;

  @$core.override
  $pb.BuilderInfo get info_ => _i;

  @$core.pragma('dart2js:noInline')
  static ControlAck create() => ControlAck._();
  @$core.override
  ControlAck createEmptyInstance() => create();
  @$core.pragma('dart2js:noInline')
  static ControlAck getDefault() => _defaultInstance ??=
      $pb.GeneratedMessage.$_defaultFor<ControlAck>(create);
  static ControlAck? _defaultInstance;

  @$pb.TagNumber(1)
  $fixnum.Int64 get frameId => $_getI64(0);
  @$pb.TagNumber(1)
  set frameId($fixnum.Int64 value) => $_setInt64(0, value);
  @$pb.TagNumber(1)
  $core.bool hasFrameId() => $_has(0);
  @$pb.TagNumber(1)
  void clearFrameId() => $_clearField(1);

  @$pb.TagNumber(2)
  $core.bool get following => $_getBF(1);
  @$pb.TagNumber(2)
  set following($core.bool value) => $_setBool(1, value);
  @$pb.TagNumber(2)
  $core.bool hasFollowing() => $_has(1);
  @$pb.TagNumber(2)
  void clearFollowing() => $_clearField(2);

  @$pb.TagNumber(3)
  $core.String get status => $_getSZ(2);
  @$pb.TagNumber(3)
  set status($core.String value) => $_setString(2, value);
  @$pb.TagNumber(3)
  $core.bool hasStatus() => $_has(2);
  @$pb.TagNumber(3)
  void clearStatus() => $_clearField(3);
}

const $core.bool _omitFieldNames =
    $core.bool.fromEnvironment('protobuf.omit_field_names');
const $core.bool _omitMessageNames =
    $core.bool.fromEnvironment('protobuf.omit_message_names');
