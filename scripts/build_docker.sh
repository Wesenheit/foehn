docker buildx build \
  --platform linux/arm64 \
  --build-arg BASE_IMAGE=quay.io/pypa/manylinux_2_28_aarch64 \
  --load \
  -f Dockerfile.pmix \
  -t ghcr.io/wesenheit/manylinux-pmix:5.0.9-aarch64 \
  .

docker buildx build \
  --platform linux/amd64 \
  --build-arg BASE_IMAGE=quay.io/pypa/manylinux_2_28_x86_64 \
  --load \
  -f Dockerfile.pmix \
  -t ghcr.io/wesenheit/manylinux-pmix:5.0.9-x86_64 \
  .
