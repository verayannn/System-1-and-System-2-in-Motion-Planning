#!/usr/bin/env bash
# One-command setup: create the virtualenv, install dependencies, build acados.
#
# Usage:
#   ./setup.sh                 # full setup
#   ./setup.sh --jobs 8        # extra arguments go to script/setup_acados.py
#   VENV_DIR=/tmp/env ./setup.sh
#
# Afterwards only 'source .venv/bin/activate' is needed: the acados environment
# is installed into the virtualenv, so nothing else has to be sourced.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

VENV_DIR="${VENV_DIR:-$REPO_ROOT/.venv}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"

info() { printf '\n[setup] %s\n' "$1"; }
fail() { printf '\n[error] %s\n' "$1" >&2; exit 1; }

require_tool() {
  command -v "$1" >/dev/null 2>&1 || fail "$1 is required but was not found. $2"
}

case "$(uname -s)" in
  Darwin) INSTALL_HINT="Install it with: xcode-select --install && brew install cmake" ;;
  *)      INSTALL_HINT="Install it with: sudo apt-get install -y build-essential cmake python3-dev" ;;
esac

require_tool cmake "$INSTALL_HINT"
require_tool make "$INSTALL_HINT"

info "Creating the Python environment in $VENV_DIR"
if [ -x "$VENV_DIR/bin/python" ]; then
  echo "Reusing the existing environment."
elif command -v uv >/dev/null 2>&1; then
  uv venv --python "$PYTHON_VERSION" "$VENV_DIR"
elif command -v "python$PYTHON_VERSION" >/dev/null 2>&1; then
  "python$PYTHON_VERSION" -m venv "$VENV_DIR"
else
  fail "Neither uv nor python$PYTHON_VERSION was found. Install uv (https://docs.astral.sh/uv/) or Python $PYTHON_VERSION."
fi

VENV_PYTHON="$VENV_DIR/bin/python"

info "Installing Python dependencies"
if command -v uv >/dev/null 2>&1 && [ "$VENV_DIR" = "$REPO_ROOT/.venv" ]; then
  # --inexact keeps the acados_template install, which is not in the lock file.
  uv sync --extra acados-template --inexact
else
  "$VENV_PYTHON" -m ensurepip --upgrade >/dev/null 2>&1 || true
  "$VENV_PYTHON" -m pip install --upgrade pip
  "$VENV_PYTHON" -m pip install -r requirements.txt
  "$VENV_PYTHON" -m pip install -e .
fi

info "Building and registering acados"
"$VENV_PYTHON" script/setup_acados.py "$@"

info "Setup complete. Activate the environment with:"
echo "  source ${VENV_DIR#$REPO_ROOT/}/bin/activate"
