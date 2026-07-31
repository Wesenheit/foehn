#!/usr/bin/env bash
set -euxo pipefail

PMIX_VERSION="5.0.9"
PMIX_PREFIX="/opt/pmix"
BUILD_DIR="/tmp/pmix-build"

echo ">>> Building PMIx v${PMIX_VERSION}"

# Clean up any previous attempt
rm -rf "${BUILD_DIR}"
mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

if command -v apt-get &>/dev/null; then
    apt-get update
    apt-get install -y --no-install-recommends libevent-dev libhwloc-dev
elif command -v yum &>/dev/null; then
    yum install -y libevent-devel hwloc-devel
elif command -v apk &>/dev/null; then
    apk add --no-cache libevent-dev hwloc-dev
else
    echo "ERROR: No supported package manager found"
    exit 1
fi

# Download the official release tarball (no autogen.pl needed)
curl -fsSL \
    "https://github.com/openpmix/openpmix/releases/download/v${PMIX_VERSION}/pmix-${PMIX_VERSION}.tar.bz2" \
    -o "pmix-${PMIX_VERSION}.tar.bz2"

tar -xjf "pmix-${PMIX_VERSION}.tar.bz2"
cd "pmix-${PMIX_VERSION}"

./configure \
    --prefix="${PMIX_PREFIX}" \
    --enable-shared \
    --disable-static \
    --disable-server \
    --disable-tools \
    --disable-pmi-backward-compat \
    --disable-sphinx \

make -j4
make install


echo ">>> PMIx v${PMIX_VERSION} installed to ${PMIX_PREFIX}"
