#!/usr/bin/env bash
set -euo pipefail

# Node-local runtime; benchmark inputs and outputs remain on shared storage.
runtime=/var/tmp/rvln-isaac6-container
shared=/mnt/workspace/nop/rvln
mkdir -p "$runtime/home" "$runtime/cache" "$runtime/uv-cache"

entrypoint=$1
shift
network_args=(--shm-size=4g)
if [[ ${RVLN_ISAAC_ROS_HOST:-0} == 1 ]]; then
  network_args=(--network=host --shm-size=4g)
fi
name_args=()
if [[ -n ${RVLN_ISAAC_CONTAINER_NAME:-} ]]; then
  name_args=(--name "$RVLN_ISAAC_CONTAINER_NAME")
fi
exec docker run --rm --runtime=runc "${network_args[@]}" "${name_args[@]}" \
  --user 2001:2001 \
  --device=/dev/nvidia0 --device=/dev/nvidiactl \
  --device=/dev/nvidia-uvm --device=/dev/nvidia-modeset \
  --device=/dev/dri/renderD128 --device=/dev/dri/card0 \
  --mount type=bind,source=/usr/lib/x86_64-linux-gnu,target=/host-nvidia,readonly \
  --mount type=bind,source=/var/tmp/rvln-isaac6-driver-libs,target=/driver-libs,readonly \
  --mount type=bind,source=/usr/share/vulkan/icd.d,target=/host-vulkan-icd,readonly \
  --mount type=bind,source=/usr/share/glvnd/egl_vendor.d,target=/host-egl-vendor,readonly \
  --mount type=bind,source="$shared",target="$shared" \
  --mount type=bind,source="$runtime",target="$runtime" \
  --mount type=bind,source=/home/nop/.local/bin/uv,target=/usr/local/bin/uv,readonly \
  --workdir="$shared/bench" \
  -e HOME="$runtime/home" \
  -e USER=nop -e LOGNAME=nop \
  -e XDG_CACHE_HOME="$runtime/cache" \
  -e UV_CACHE_DIR="$runtime/uv-cache" \
  -e UV_PROJECT_ENVIRONMENT="$runtime/venv" \
  -e UV_PYTHON_INSTALL_DIR="$shared/.uv-python" \
  -e UV_LINK_MODE=hardlink -e UV_HTTP_TIMEOUT=300 \
  -e OMNI_KIT_ACCEPT_EULA=YES \
  -e ROS_DISTRO=humble \
  -e ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}" \
  -e FASTRTPS_DEFAULT_PROFILES_FILE="$shared/bench/scripts/fastdds_udp.xml" \
  -e LD_LIBRARY_PATH="$runtime/venv/lib/python3.12/site-packages/isaacsim/exts/isaacsim.ros2.core/humble/lib:/driver-libs" \
  -e VK_ICD_FILENAMES=/host-vulkan-icd/nvidia_icd.json \
  -e __EGL_VENDOR_LIBRARY_FILENAMES=/host-egl-vendor/10_nvidia.json \
  --entrypoint "$entrypoint" rvln-isaac6-ubuntu22 "$@"
