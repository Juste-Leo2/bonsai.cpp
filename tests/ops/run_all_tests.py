#!/usr/bin/env python3
"""Runner pour tous les tests unitaires des opérations du diffuseur.

Usage:
    python tests/ops/run_all_tests.py

Ce script exécute séquentiellement tous les tests d'opérations pour s'assurer
que chaque partie du diffuseur fonctionne correctement avant l'intégration.
"""

import subprocess
import sys
from pathlib import Path

def run_command(cmd, description):
    """Exécute une commande et affiche le résultat."""
    print(f"\n{'='*60}")
    print(f"  {description}")
    print(f"{'='*60}")
    print(f"  Command: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=True, text=True)

    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)

    if result.returncode != 0:
        print(f"FAIL: {description} (exit code: {result.returncode})")
        return False
    else:
        print(f"PASS: {description}")
        return True


def main():
    project_root = Path(__file__).resolve().parent.parent.parent
    build_dir = project_root / "build"

    # 1. Build des tests C
    print(f"{'='*60}")
    print("  Building C test binaries...")
    print(f"{'='*60}")

    build_result = subprocess.run(
        ["cmake", "--build", str(build_dir), "--target", "test_b1_linear"],
        capture_output=True,
        text=True,
    )
    if build_result.returncode != 0:
        print("Build failed:")
        print(build_result.stdout)
        print(build_result.stderr)
        return 1

    print("Build successful!")

    # 2. Exécution des tests Python
    all_passed = True

    test_files = [
        "tests/ops/test_b1_linear.py",
        "tests/ops/test_rope_2d.py",
        "tests/ops/test_rms_norm_qk.py",
        "tests/ops/test_flash_attention.py",
        "tests/ops/test_mlp_gated.py",
        "tests/ops/test_modulation.py",
        "tests/c_ops/test_b1_linear.py",
    ]

    for test_file in test_files:
        test_path = project_root / test_file
        if not test_path.exists():
            print(f"WARNING: {test_file} not found, skipping")
            continue

        success = run_command(
            ["python", "-m", "pytest", str(test_path), "-v"],
            f"Testing {test_file}"
        )
        if not success:
            all_passed = False

    # Résumé final
    print(f"\n{'='*60}")
    if all_passed:
        print("  ALL TESTS PASSED!")
    else:
        print("  SOME TESTS FAILED!")
    print(f"{'='*60}")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
