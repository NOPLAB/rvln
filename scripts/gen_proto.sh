#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${REPO_ROOT}/src/raspicat_vla_proto/raspicat_vla_proto"
PROTO_DIR="${REPO_ROOT}/proto"

mkdir -p "${OUT_DIR}"

python3 -m grpc_tools.protoc \
    -I "${PROTO_DIR}" \
    --python_out="${OUT_DIR}" \
    --grpc_python_out="${OUT_DIR}" \
    "${PROTO_DIR}/edge_action.proto"

# grpc_tools generates `import <name>_pb2` -- rewrite to relative.
sed -i 's/^import edge_action_pb2/from . import edge_action_pb2/' "${OUT_DIR}/edge_action_pb2_grpc.py"

echo "Generated:"
ls -1 "${OUT_DIR}"/edge_action_pb2*.py

# Dart stubs for the inference app (proto/edge_action.proto only). Optional:
# needs the Flutter/Dart toolchain + `dart pub global activate protoc_plugin`.
# Generated files are committed (the Flutter build has no protoc step).
DART_PLUGIN="${HOME}/.pub-cache/bin/protoc-gen-dart"
DART_OUT="${REPO_ROOT}/app/inference/lib/src/grpc/gen"
if [[ -x "${DART_PLUGIN}" ]]; then
    mkdir -p "${DART_OUT}"
    python3 -m grpc_tools.protoc \
        -I "${PROTO_DIR}" \
        --plugin=protoc-gen-dart="${DART_PLUGIN}" \
        --dart_out=grpc:"${DART_OUT}" \
        "${PROTO_DIR}/edge_action.proto"
    echo "Generated (Dart):"
    ls -1 "${DART_OUT}"/edge_action.*.dart
else
    echo "protoc-gen-dart not found (${DART_PLUGIN}); skipping Dart stubs." >&2
    echo "Install with: dart pub global activate protoc_plugin" >&2
fi
