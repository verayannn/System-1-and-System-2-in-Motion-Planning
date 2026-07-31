"""Build the repo-local acados backend and register it inside the active venv.

Besides compiling acados, this installs the template renderer and writes the
acados environment into the virtualenv itself, so no shell command has to
export anything before running MPC.
"""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path

ENV_BLOCK_START = "# >>> acados environment (managed by script/setup_acados.py) >>>"
ENV_BLOCK_END = "# <<< acados environment (managed by script/setup_acados.py) <<<"


def run(cmd: list[str], *, cwd: Path, env: dict[str, str] | None = None) -> None:
    print("[cmd]", " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd), env=env, check=True)


def is_git_checkout(path: Path) -> bool:
    """Return whether ``path`` owns a Git worktree, not just a parent does."""
    return (path / ".git").exists()


def check_acados_sources(acados_root: Path) -> None:
    required = [
        acados_root / "external" / "blasfeo" / "CMakeLists.txt",
        acados_root / "external" / "blasfeo" / "include" / "blasfeo.h",
        acados_root / "external" / "hpipm" / "CMakeLists.txt",
        acados_root / "external" / "hpipm" / "include" / "hpipm_common.h",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise SystemExit(
            "acados source tree is incomplete. Missing:\n  "
            + "\n  ".join(missing)
            + "\nThe repository vendors acados and these files are tracked by the parent "
            "repository. Update to the current repository revision with:\n"
            "  git pull --rebase"
        )


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


def tera_binary(acados_root: Path) -> Path:
    return acados_root / "bin" / ("t_renderer.exe" if os.name == "nt" else "t_renderer")


def tera_download_url(*, cwd: Path, env: dict[str, str]) -> str | None:
    """Ask acados which renderer build matches this platform."""
    code = (
        "from acados_template.utils import ("
        "TERA_DEFAULT_VERSION, PLATFORM2TERA, get_binary_ext, get_architecture_amd64_arm64)\n"
        "import sys\n"
        "v = TERA_DEFAULT_VERSION\n"
        "print('https://github.com/acados/tera_renderer/releases/download/'\n"
        "      f'v{v}/t_renderer-v{v}-{PLATFORM2TERA[sys.platform]}-'\n"
        "      f'{get_architecture_amd64_arm64()}{get_binary_ext()}')\n"
    )
    probe = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(cwd),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if probe.returncode != 0:
        return None
    return probe.stdout.strip() or None


def tera_manual_instructions(acados_root: Path, url: str | None) -> str:
    target = url or "https://github.com/acados/tera_renderer/releases"
    return (
        "Could not install the acados template renderer automatically.\n"
        "Download it manually, then re-run this script:\n"
        f"  curl -fL -o {tera_binary(acados_root)} {target}\n"
        f"  chmod +x {tera_binary(acados_root)}"
    )


def download_tera_with_curl(url: str, tera_path: Path) -> bool:
    """Fall back to curl, which uses the system trust store.

    Python builds on macOS frequently ship without root certificates, so the
    download inside acados fails with a certificate error even when the network
    is fine.
    """
    curl = shutil.which("curl")
    if curl is None:
        return False
    tmp_path = tera_path.with_suffix(tera_path.suffix + ".download")
    try:
        run([curl, "-fL", "--retry", "3", "-o", str(tmp_path), url], cwd=tera_path.parent)
    except subprocess.CalledProcessError:
        tmp_path.unlink(missing_ok=True)
        return False
    tmp_path.replace(tera_path)
    return True


def ensure_tera_renderer(acados_root: Path, *, cwd: Path, source: Path | None = None) -> Path:
    """Install the renderer acados uses to generate solver code.

    ``acados_template`` asks on stdin when this binary is missing. Benchmark
    workers have no stdin, so the prompt surfaces as an EOFError and every MPC
    solve fails with an unrelated-looking error instead of solving.
    """
    tera_path = tera_binary(acados_root)
    if tera_path.is_file() and os.access(tera_path, os.X_OK):
        return tera_path

    tera_path.parent.mkdir(parents=True, exist_ok=True)
    if source is not None:
        shutil.copy2(source, tera_path)
        tera_path.chmod(0o755)
        return tera_path

    env = dict(os.environ)
    env["ACADOS_SOURCE_DIR"] = str(acados_root)
    url = tera_download_url(cwd=cwd, env=env)
    print(f"[info] installing the acados template renderer from {url or 'the acados release page'}")
    attempt = subprocess.run(
        [sys.executable, "-c", "from acados_template.utils import get_tera; get_tera(force_download=True)"],
        cwd=str(cwd),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    if not (attempt.returncode == 0 and tera_path.is_file()):
        if url is None or not download_tera_with_curl(url, tera_path):
            print(attempt.stderr.strip())
            raise SystemExit(tera_manual_instructions(acados_root, url))
    tera_path.chmod(0o755)
    return tera_path


def venv_dir() -> Path | None:
    """Return the active virtualenv, or None when running against a base install."""
    if sys.prefix == getattr(sys, "base_prefix", sys.prefix):
        return None
    return Path(sys.prefix)


def install_venv_hook(acados_root: Path) -> Path | None:
    """Export the acados environment from every interpreter in this venv.

    A ``.pth`` file keeps the settings with the environment instead of with one
    shell session, so scripts launched without sourcing anything still find
    acados.
    """
    if venv_dir() is None:
        print("[warn] not running inside a virtualenv; skipping venv environment install")
        return None

    site_packages = Path(sysconfig.get_paths()["purelib"])
    site_packages.mkdir(parents=True, exist_ok=True)
    module_path = site_packages / "_acados_env.py"
    module_path.write_text(
        '"""Acados environment for this virtualenv, written by script/setup_acados.py."""\n'
        "\n"
        "import os\n"
        "import sys\n"
        "from pathlib import Path\n"
        "\n"
        f"ACADOS_ROOT = Path(r{str(acados_root)!r})\n"
        "\n"
        "\n"
        "def _prepend(key, value):\n"
        "    parts = [p for p in os.environ.get(key, '').split(os.pathsep) if p]\n"
        "    if value not in parts:\n"
        "        os.environ[key] = os.pathsep.join([value, *parts])\n"
        "\n"
        "\n"
        "def configure():\n"
        "    if not ACADOS_ROOT.is_dir():\n"
        "        return\n"
        "    os.environ.setdefault('ACADOS_SOURCE_DIR', str(ACADOS_ROOT))\n"
        "    os.environ.setdefault('ACADOS_INSTALL_DIR', str(ACADOS_ROOT))\n"
        "    os.environ.setdefault(\n"
        "        'ACADOS_PYTHON_INTERFACE_PATH',\n"
        "        str(ACADOS_ROOT / 'interfaces' / 'acados_template'),\n"
        "    )\n"
        "    renderer = ACADOS_ROOT / 'bin' / ('t_renderer.exe' if os.name == 'nt' else 't_renderer')\n"
        "    if renderer.is_file():\n"
        "        os.environ.setdefault('TERA_PATH', str(renderer))\n"
        "    lib_dir = ACADOS_ROOT / 'lib'\n"
        "    if lib_dir.is_dir():\n"
        "        keys = (\n"
        "            ('DYLD_LIBRARY_PATH', 'DYLD_FALLBACK_LIBRARY_PATH')\n"
        "            if sys.platform == 'darwin'\n"
        "            else ('LD_LIBRARY_PATH',)\n"
        "        )\n"
        "        for key in keys:\n"
        "            _prepend(key, str(lib_dir))\n"
        "\n"
        "\n"
        "configure()\n"
    )

    pth_path = site_packages / "zz_acados_env.pth"
    pth_path.write_text("import _acados_env\n")
    return module_path


def patch_activate_script(acados_root: Path) -> Path | None:
    """Export the same variables to the shell when the venv is activated.

    The dynamic loader reads its search path once at process start, so the
    shell-level export is what lets compilers and freshly spawned processes
    link against the local acados build.
    """
    venv = venv_dir()
    if venv is None:
        return None
    activate = venv / ("Scripts/activate" if os.name == "nt" else "bin/activate")
    if not activate.is_file():
        print(f"[warn] no activation script to patch: {activate}")
        return None

    lib_var = "DYLD_LIBRARY_PATH" if platform.system().lower() == "darwin" else "LD_LIBRARY_PATH"
    block = [
        ENV_BLOCK_START,
        f'export ACADOS_SOURCE_DIR="{acados_root}"',
        'export ACADOS_INSTALL_DIR="$ACADOS_SOURCE_DIR"',
        'export ACADOS_PYTHON_INTERFACE_PATH="$ACADOS_SOURCE_DIR/interfaces/acados_template"',
        'export TERA_PATH="$ACADOS_SOURCE_DIR/bin/t_renderer"',
        f'export {lib_var}="$ACADOS_SOURCE_DIR/lib:${{{lib_var}:-}}"',
    ]
    if platform.system().lower() == "darwin":
        block.append('export DYLD_FALLBACK_LIBRARY_PATH="$ACADOS_SOURCE_DIR/lib:${DYLD_FALLBACK_LIBRARY_PATH:-}"')
    block.append(ENV_BLOCK_END)

    lines = activate.read_text().splitlines()
    kept: list[str] = []
    inside = False
    for line in lines:
        if line.strip() == ENV_BLOCK_START:
            inside = True
            continue
        if line.strip() == ENV_BLOCK_END:
            inside = False
            continue
        if not inside:
            kept.append(line)
    while kept and not kept[-1].strip():
        kept.pop()
    activate.write_text("\n".join([*kept, "", *block, ""]))
    return activate


def verify_installation(repo_root: Path, acados_root: Path) -> None:
    tera_path = tera_binary(acados_root)
    if not (tera_path.is_file() and os.access(tera_path, os.X_OK)):
        raise SystemExit(tera_manual_instructions(acados_root, None))
    code = (
        "import acados_template, sofai_tool, safe_control\n"
        "from solvers._s2_common import detect_acados_root\n"
        "from solvers.S2_mpc import solve_MPC_with_info\n"
        "assert detect_acados_root() is not None, 'acados shared libraries were not found'\n"
        "print('[ok] imports: acados_template, sofai_tool, safe_control, S2 MPC')\n"
    )
    # A clean environment proves the venv itself carries the settings, rather
    # than inheriting them from the shell that ran this script.
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("ACADOS_", "TERA_", "LD_LIBRARY_PATH", "DYLD_"))
    }
    run([sys.executable, "-c", code], cwd=repo_root, env=env)


def write_env_file(repo_root: Path, acados_root: Path) -> Path:
    env_path = repo_root / ".env.acados"
    lib_var = "DYLD_LIBRARY_PATH" if platform.system().lower() == "darwin" else "LD_LIBRARY_PATH"
    lines = [
        "# Optional: the virtualenv already exports these after 'source .venv/bin/activate'.",
        "# Source this file to get the same settings in a shell that is not using the venv.",
        f'export ACADOS_SOURCE_DIR="{acados_root}"',
        f'export ACADOS_INSTALL_DIR="{acados_root}"',
        f'export ACADOS_PYTHON_INTERFACE_PATH="$ACADOS_SOURCE_DIR/interfaces/acados_template"',
        'export TERA_PATH="$ACADOS_SOURCE_DIR/bin/t_renderer"',
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
    p.add_argument("--skip_submodules", action="store_true", help="Do not update nested submodules when using a standalone acados checkout.")
    p.add_argument("--tera_binary", type=Path, default=None, help="Install this t_renderer binary instead of downloading one.")
    p.add_argument("--skip_verify", action="store_true", help="Do not run the post-installation import check.")
    args = p.parse_args()

    acados_root = args.acados_root.expanduser().resolve()
    if not (acados_root / "CMakeLists.txt").is_file():
        raise SystemExit(f"acados root does not look valid: {acados_root}")

    # The repository ships a complete, vendored acados source tree. Its
    # historical .gitmodules file is data from upstream, not an instruction to
    # run Git from this non-worktree directory. A user-provided acados checkout
    # can still populate its own nested submodules.
    if not args.skip_submodules and is_git_checkout(acados_root):
        run(["git", "submodule", "sync", "--recursive"], cwd=acados_root)
        run(["git", "submodule", "update", "--init", "--recursive"], cwd=acados_root)
    check_acados_sources(acados_root)

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
        # The vendored BLASFEO snapshot intentionally excludes upstream's
        # ignored ISA-test assembly files. GENERIC is portable across macOS and
        # Linux and avoids CMake probing those unavailable files.
        run(
            [
                "cmake",
                "-DACADOS_WITH_QPOASES=OFF",
                "-DBUILD_SHARED_LIBS=ON",
                "-DBLASFEO_TARGET=GENERIC",
                "-DBLASFEO_EXAMPLES=OFF",
                "..",
            ],
            cwd=build_dir,
        )
        run(["cmake", "--build", ".", "--target", "install", "-j", str(args.jobs)], cwd=build_dir)

    check_libs(acados_root)
    install_editable(acados_root / "interfaces" / "acados_template", cwd=repo_root)
    tera_path = ensure_tera_renderer(acados_root, cwd=repo_root, source=args.tera_binary)
    hook_path = install_venv_hook(acados_root)
    activate_path = patch_activate_script(acados_root)
    env_path = write_env_file(repo_root, acados_root)

    if not args.skip_verify:
        verify_installation(repo_root, acados_root)

    print(f"[ok] acados root: {acados_root}")
    print(f"[ok] template renderer: {tera_path}")
    if hook_path is not None:
        print(f"[ok] venv environment hook: {hook_path}")
    if activate_path is not None:
        print(f"[ok] patched activation script: {activate_path}")
    print(f"[ok] wrote: {env_path}")
    if hook_path is None:
        print(f"[next] source {env_path}")
    else:
        print("[next] nothing to source: the environment is active in this venv")


if __name__ == "__main__":
    main()
