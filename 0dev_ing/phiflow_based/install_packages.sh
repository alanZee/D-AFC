#!/usr/bin/env bash
# 安装脚本 (WSL / Ubuntu)
# 用途：在指定 conda 环境中以 editable 模式安装仓库内的包： myafc。
# 用法：在 WSL 中运行：
    # chmod +x install_packages.sh
    # ./install_packages.sh -e zero_phi310
# 可选：-n/--dry-run 仅打印将要执行的操作而不实际安装。

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="zero_phi310"
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    -e|--env)
      ENV_NAME="$2"
      shift; shift
      ;;
    -n|--dry-run)
      DRY_RUN=1
      shift
      ;;
    -p|--paths)
      # 传入多个路径请用引号并以逗号分隔
      IFS=',' read -ra PARR <<< "$2"
      PACKAGE_DIRS=("${PARR[@]}")
      shift; shift
      ;;
    -h|--help)
      echo "Usage: $0 [-e env_name] [-n] [-p \"/abs/path1,/abs/path2\"]"
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2; exit 1
      ;;
  esac
done

# 默认包目录（基于仓库根目录）
PACKAGE_DIRS=(
  "$SCRIPT_DIR/0mydev/myafc/src"
)

echo "Script dir: $SCRIPT_DIR"
echo "Conda env: $ENV_NAME"
echo "Package dirs:"
for p in "${PACKAGE_DIRS[@]}"; do echo "  - $p"; done
if [[ $DRY_RUN -eq 1 ]]; then
  echo "Dry run mode - nothing will be installed."; exit 0
fi

# Ensure conda is initialised in this shell
if command -v conda >/dev/null 2>&1; then
  echo "Using conda from PATH"
else
  if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
  elif [[ -f "/opt/conda/etc/profile.d/conda.sh" ]]; then
    source "/opt/conda/etc/profile.d/conda.sh"
  else
    echo "Warning: conda not found in PATH and common locations. Make sure conda is available or initialize it before running this script." >&2
  fi
fi

# Activate the conda environment
if conda info --envs | grep -q "^$ENV_NAME"; then
  echo "Activating conda env: $ENV_NAME"
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  # Temporarily disable nounset to avoid failures when activation scripts reference
  # variables that may be unset (e.g. MKL_INTERFACE_LAYER in activate.d scripts).
  set +u
  conda activate "$ENV_NAME"
  set -u
else
  echo "Conda environment '$ENV_NAME' not found in 'conda info --envs'. Attempting to activate anyway..."
  source "$(conda info --base)/etc/profile.d/conda.sh" || true
  set +u
  conda activate "$ENV_NAME" || true
  set -u
fi

# Upgrade pip
echo "Upgrading pip, setuptools, wheel..."
python -m pip install --upgrade pip setuptools wheel

# Install packages editable
CREATED=()

# Cleanup function to remove any temporary pyproject.toml files (runs on EXIT)
cleanup() {
  if [[ ${#CREATED[@]} -gt 0 ]]; then
    echo "Cleaning up temporary pyproject.toml files..."
    for f in "${CREATED[@]}"; do
      echo "  - $f"
      rm -f "$f" || echo "Failed to remove $f"
    done
  fi
}
trap cleanup EXIT

for p in "${PACKAGE_DIRS[@]}"; do
  if [[ ! -d "$p" ]]; then
    echo "Path not found, skipping: $p" >&2
    continue
  fi
  # Determine package root (if caller passed .../src, use its parent as the project root)
  pkgRoot="$p"
  if [[ "$pkgRoot" == */src ]]; then
    pkgRoot=$(dirname "$pkgRoot")
  fi

  # If neither pyproject.toml nor setup.py exist at project root, create a minimal pyproject.toml
  if [[ ! -f "$pkgRoot/pyproject.toml" && ! -f "$pkgRoot/setup.py" ]]; then
    echo "pyproject.toml not found in $pkgRoot — creating minimal pyproject.toml"
    cat > "$pkgRoot/pyproject.toml" <<'PY'
[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "${NAME_REPLACED}"
version = "0.0.1"
description = "Local package"
readme = "README.md"
license = {text = "MIT"}

[tool.setuptools]
package-dir = {"" = "src"}

[tool.setuptools.packages.find]
where = ["src"]
PY
    # Patch the placeholder name with the actual folder name
    pkgName=$(basename "$pkgRoot")
  sed -i "s/\${NAME_REPLACED}/$pkgName/" "$pkgRoot/pyproject.toml"
  echo "Wrote: $pkgRoot/pyproject.toml"
  CREATED+=("$pkgRoot/pyproject.toml")
  fi

  echo "Installing editable: $pkgRoot"
  python -m pip install -e "$pkgRoot"
done

# Remove any temporary pyproject.toml files we created
if [[ ${#CREATED[@]} -gt 0 ]]; then
  echo "Removing temporary pyproject.toml files:"
  for f in "${CREATED[@]}"; do
    echo "  - $f"
    rm -f "$f" || echo "Failed to remove $f"
  done
fi

# Verification imports
echo "Verifying imports: myafc"
PY_CODE='import importlib, sys; name=sys.argv[1];
try:
    m=importlib.import_module(name)
    print("OK", name, getattr(m, "__file__", "<package>"))
except Exception as e:
    print("FAIL", name, e)'

for mod in myafc; do
  echo "- testing import: $mod"
  python - <<PYBLOCK
import importlib, sys
name='$mod'
try:
    m=importlib.import_module(name)
    print('OK', name, getattr(m, '__file__', '<package>'))
except Exception as e:
    print('FAIL', name, e)
PYBLOCK
done

echo "Done. If imports failed, ensure each package directory contains a proper Python package (a top-level folder with __init__.py) or a setup.py/pyproject.toml for pip to install." 
