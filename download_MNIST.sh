#!/usr/bin/env bash
# Downloads the MNIST dataset as .gz files (no decompression).
# Usage: ./download_mnist.sh [target_dir]     (default: ../MNIST/MNIST/raw)
# Needs: bash, and either curl or wget, plus gzip (standard on Linux/macOS/WSL/Git Bash).

set -euo pipefail

# Default matches the paths used by the C++ code: ../MNIST/MNIST/raw/<file>.gz
TARGET_DIR="${1:-MNIST/MNIST/raw}"

# Mirrors are tried in order (the original yann.lecun.com host is often down)
MIRRORS=(
  "https://ossci-datasets.s3.amazonaws.com/mnist"
  "https://storage.googleapis.com/cvdf-datasets/mnist"
  "https://yann.lecun.com/exdb/mnist"
)

FILES=(
  "train-images-idx3-ubyte.gz"   # 60 000 training images
  "train-labels-idx1-ubyte.gz"   # 60 000 training labels
  "t10k-images-idx3-ubyte.gz"    # 10 000 test images
  "t10k-labels-idx1-ubyte.gz"    # 10 000 test labels
)

# Pick a download tool
if command -v curl >/dev/null 2>&1; then
  download() { curl -fL --retry 3 -o "$2" "$1"; }
elif command -v wget >/dev/null 2>&1; then
  download() { wget -q -O "$2" "$1"; }
else
  echo "Error: neither curl nor wget found. Please install one of them." >&2
  exit 1
fi

mkdir -p "$TARGET_DIR"

for name in "${FILES[@]}"; do
  out="$TARGET_DIR/$name"

  if [[ -f "$out" ]] && gzip -t "$out" 2>/dev/null; then
    echo "[skip] $name already exists"
    continue
  fi

  success=0
  for mirror in "${MIRRORS[@]}"; do
    echo "[get ] $name  <-  $mirror"
    # gzip -t verifies the download is a valid, uncorrupted gzip file
    if download "$mirror/$name" "$out" && gzip -t "$out"; then
      success=1
      break
    fi
    rm -f "$out"
    echo "       failed, trying next mirror..."
  done

  if [[ $success -ne 1 ]]; then
    echo "Error: could not download $name from any mirror." >&2
    exit 1
  fi
done

echo "Done. MNIST .gz files are in: $TARGET_DIR"