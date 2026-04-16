"""Environment validation for Apple Silicon ML development."""

import sys
import platform
import subprocess


def validate_environment() -> dict:
    """Check Python version, MLX availability, Metal GPU status, and memory size.

    Returns a dict with keys:
        python_version: str — current Python version
        python_ok: bool — True if Python >= 3.13
        mlx_available: bool — True if mlx can be imported
        mlx_version: str | None — mlx version string
        metal_available: bool — True if Metal GPU is available
        memory_gb: float — total system memory in GB
        memory_ok: bool — True if memory >= 16 GB
        chip: str — Apple Silicon chip name (e.g. "Apple M4 Pro")
        platform: str — e.g. "macOS-15.x-arm64"
    """
    results: dict = {}

    # Python version
    ver = sys.version_info
    results["python_version"] = f"{ver.major}.{ver.minor}.{ver.micro}"
    results["python_ok"] = (ver.major, ver.minor) >= (3, 13)

    # MLX availability
    try:
        import mlx.core as mx

        results["mlx_available"] = True
        results["mlx_version"] = mx.__version__
    except ImportError:
        results["mlx_available"] = False
        results["mlx_version"] = None

    # Metal GPU status
    try:
        import mlx.core as mx

        results["metal_available"] = mx.metal.is_available()
    except Exception:
        results["metal_available"] = False

    # System memory
    try:
        mem_bytes = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"]).strip())
        results["memory_gb"] = round(mem_bytes / (1024**3), 1)
    except Exception:
        results["memory_gb"] = 0.0
    results["memory_ok"] = results["memory_gb"] >= 16

    # Chip info
    try:
        chip = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"]
        ).decode().strip()
        results["chip"] = chip
    except Exception:
        results["chip"] = "Unknown"

    results["platform"] = platform.platform()

    return results


def print_environment_report() -> dict:
    """Validate environment and print a formatted report. Returns the results dict."""
    results = validate_environment()

    def status(ok: bool) -> str:
        return "✅" if ok else "❌"

    print("=" * 50)
    print("  Environment Validation Report")
    print("=" * 50)
    print(f"  Platform : {results['platform']}")
    print(f"  Chip     : {results['chip']}")
    print(f"  Python   : {results['python_version']}  {status(results['python_ok'])}")
    print(f"  MLX      : {results.get('mlx_version', 'not installed')}  {status(results['mlx_available'])}")
    print(f"  Metal GPU: {'available' if results['metal_available'] else 'NOT available'}  {status(results['metal_available'])}")
    print(f"  Memory   : {results['memory_gb']} GB  {status(results['memory_ok'])}")
    print("=" * 50)

    if not results["metal_available"]:
        print("\n⚠️  Metal GPU not available. Computation will fall back to CPU.")
        print("    Ensure you are on macOS 14+ with Apple Silicon and latest MLX.")

    return results
