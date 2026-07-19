from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str], *, cwd: Path) -> None:
    print("[cmd]", " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd), check=True)


def install_editable(package_dir: Path, *, cwd: Path) -> None:
    has_pip = subprocess.run(
        [sys.executable, "-m", "pip", "--version"],
        cwd=str(cwd),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    ).returncode == 0
    if has_pip:
        run([sys.executable, "-m", "pip", "install", "-e", str(package_dir)], cwd=cwd)
        return

    uv = shutil.which("uv")
    if uv:
        run([uv, "pip", "install", "--python", sys.executable, "-e", str(package_dir)], cwd=cwd)
        return

    run([sys.executable, "-m", "ensurepip", "--upgrade"], cwd=cwd)
    run([sys.executable, "-m", "pip", "install", "-e", str(package_dir)], cwd=cwd)


def lib_patterns() -> tuple[str, ...]:
    if platform.system().lower() == "darwin":
        return ("libacados.dylib", "libblasfeo*.dylib", "libhpipm*.dylib")
    return ("libacados.so", "libblasfeo*.so", "libhpipm*.so")


def check_libs(acados_root: Path) -> None:
    missing = []
    lib_dir = acados_root / "lib"
    for pattern in lib_patterns():
        if not list(lib_dir.glob(pattern)):
            missing.append(str(lib_dir / pattern))
    if missing:
        raise SystemExit("Missing acados shared libraries:\n  " + "\n  ".join(missing))


def write_env_file(repo_root: Path, acados_root: Path) -> Path:
    env_path = repo_root / ".env.acados"
    lib_var = "DYLD_LIBRARY_PATH" if platform.system().lower() == "darwin" else "LD_LIBRARY_PATH"
    lines = [
        "# Source this file before running S2 MPC benchmarks.",
        f'export ACADOS_SOURCE_DIR="{acados_root}"',
        f'export ACADOS_INSTALL_DIR="{acados_root}"',
        f'export ACADOS_PYTHON_INTERFACE_PATH="$ACADOS_SOURCE_DIR/interfaces/acados_template/acados_template"',
        f'export {lib_var}="$ACADOS_SOURCE_DIR/lib:${{{lib_var}:-}}"',
    ]
    if platform.system().lower() == "darwin":
        lines.append('export DYLD_FALLBACK_LIBRARY_PATH="$ACADOS_SOURCE_DIR/lib:${DYLD_FALLBACK_LIBRARY_PATH:-}"')
    lines.append("")
    env_path.write_text("\n".join(lines))
    return env_path


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description="Build and register the repo-local acados backend.")
    p.add_argument("--acados_root", type=Path, default=repo_root / "safe_control" / "acados")
    p.add_argument("--jobs", type=int, default=max(os.cpu_count() or 2, 2))
    p.add_argument("--clean", action="store_true", help="Reconfigure CMake from scratch.")
    p.add_argument("--skip_build", action="store_true", help="Only install acados_template and write .env.acados.")
    args = p.parse_args()

    acados_root = args.acados_root.expanduser().resolve()
    if not (acados_root / "CMakeLists.txt").is_file():
        raise SystemExit(f"acados root does not look valid: {acados_root}")

    if not args.skip_build:
        build_dir = acados_root / "build"
        build_dir.mkdir(parents=True, exist_ok=True)
        if args.clean:
            for name in ("CMakeCache.txt", "CMakeFiles"):
                path = build_dir / name
                if path.is_dir():
                    subprocess.run(["cmake", "-E", "rm", "-rf", str(path)], check=True)
                elif path.exists():
                    path.unlink()
        run(["cmake", "-DACADOS_WITH_QPOASES=ON", "-DBUILD_SHARED_LIBS=ON", ".."], cwd=build_dir)
        run(["cmake", "--build", ".", "--target", "install", "-j", str(args.jobs)], cwd=build_dir)

    check_libs(acados_root)
    install_editable(acados_root / "interfaces" / "acados_template", cwd=repo_root)
    env_path = write_env_file(repo_root, acados_root)

    print(f"[ok] acados root: {acados_root}")
    print(f"[ok] wrote: {env_path}")
    print(f"[next] source {env_path}")


if __name__ == "__main__":
    main()
