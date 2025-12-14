#!/usr/bin/env python3
"""
Run IEEE 754 test packages locally.

This script handles:
1. Generating test packages if they don't exist
2. Temporarily adding test packages to the workspace
3. Running the requested tests
4. Restoring the original Nargo.toml

Usage:
  python scripts/run_tests.py                     # Run all test packages
  python scripts/run_tests.py add_shift           # Run packages matching 'add_shift'
  python scripts/run_tests.py --list              # List available test packages
  python scripts/run_tests.py --generate          # Regenerate test packages
  python scripts/run_tests.py --package ieee754_test_add_shift  # Run specific package
"""

import argparse
import glob
import os
import shutil
import subprocess
import sys
from pathlib import Path


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent


def get_test_packages(project_root: Path) -> list[str]:
    """Get list of test package directories."""
    test_packages_dir = project_root / "test_packages"
    if not test_packages_dir.exists():
        return []
    
    packages = []
    for pkg in sorted(test_packages_dir.iterdir()):
        if pkg.is_dir() and pkg.name.startswith("ieee754_"):
            packages.append(pkg.name)
    return packages


def generate_tests(project_root: Path, operation: str = None) -> bool:
    """Generate test packages."""
    print("Generating test packages...")
    
    cmd = [
        sys.executable,
        str(project_root / "scripts" / "generate_tests.py"),
        "--all",
        "--packages",
        "--output-dir", "test_packages",
        "--ci-matrix", ".github/test-matrix.json"
    ]
    
    if operation:
        cmd.extend(["--operation", operation])
    
    result = subprocess.run(cmd, cwd=project_root)
    return result.returncode == 0


def update_workspace(project_root: Path, packages: list[str]) -> str:
    """Update Nargo.toml to include test packages. Returns original content."""
    nargo_toml = project_root / "Nargo.toml"
    original_content = nargo_toml.read_text()
    
    # Build new workspace members list
    members = ['    "ieee754"', '    "ieee754_unit_tests"']
    members.extend(f'    "test_packages/{pkg}"' for pkg in packages)
    
    new_content = "[workspace]\nmembers = [\n" + ",\n".join(members) + "\n]\n"
    nargo_toml.write_text(new_content)
    
    return original_content


def restore_workspace(project_root: Path, original_content: str):
    """Restore the original Nargo.toml."""
    nargo_toml = project_root / "Nargo.toml"
    nargo_toml.write_text(original_content)


def run_tests(project_root: Path, package: str) -> bool:
    """Run tests for a specific package."""
    print(f"\n{'='*60}")
    print(f"Running tests for: {package}")
    print('='*60)
    
    result = subprocess.run(
        ["nargo", "test", "--package", package],
        cwd=project_root
    )
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(
        description="Run IEEE 754 test packages locally",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                              # Run all test packages
  %(prog)s add_shift                    # Run packages matching 'add_shift'
  %(prog)s --list                       # List available test packages
  %(prog)s --generate                   # Regenerate test packages first
  %(prog)s --generate --operation add   # Generate only addition tests
  %(prog)s --package ieee754_test_add_shift  # Run specific package by full name
        """
    )
    parser.add_argument(
        "filter",
        nargs="?",
        help="Filter packages by name (partial match)"
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List available test packages"
    )
    parser.add_argument(
        "--generate", "-g",
        action="store_true",
        help="Generate/regenerate test packages before running"
    )
    parser.add_argument(
        "--operation",
        choices=["add", "sub", "mul", "div"],
        help="When generating, only generate tests for this operation"
    )
    parser.add_argument(
        "--package", "-p",
        help="Run a specific package by full name"
    )
    parser.add_argument(
        "--keep-workspace",
        action="store_true",
        help="Don't restore original Nargo.toml after running"
    )
    
    args = parser.parse_args()
    
    project_root = get_project_root()
    
    # Generate tests if requested or if none exist
    packages = get_test_packages(project_root)
    
    if args.generate or not packages:
        if not packages:
            print("No test packages found. Generating...")
        if not generate_tests(project_root, args.operation):
            print("Failed to generate tests")
            return 1
        packages = get_test_packages(project_root)
    
    if not packages:
        print("No test packages available. Run with --generate to create them.")
        return 1
    
    # List packages if requested
    if args.list:
        print(f"Available test packages ({len(packages)}):")
        for pkg in packages:
            print(f"  {pkg}")
        return 0
    
    # Filter packages
    if args.package:
        # Exact match
        if args.package in packages:
            packages_to_run = [args.package]
        elif args.package.startswith("ieee754_"):
            packages_to_run = [p for p in packages if p == args.package]
        else:
            packages_to_run = [p for p in packages if args.package in p]
    elif args.filter:
        packages_to_run = [p for p in packages if args.filter in p]
    else:
        packages_to_run = packages
    
    if not packages_to_run:
        print(f"No packages match filter: {args.filter or args.package}")
        print("Available packages:")
        for pkg in packages[:10]:
            print(f"  {pkg}")
        if len(packages) > 10:
            print(f"  ... and {len(packages) - 10} more")
        return 1
    
    print(f"Will run {len(packages_to_run)} test package(s)")
    
    # Update workspace
    original_content = update_workspace(project_root, packages_to_run)
    
    try:
        # Run tests
        failed = []
        passed = []
        
        for pkg in packages_to_run:
            if run_tests(project_root, pkg):
                passed.append(pkg)
            else:
                failed.append(pkg)
        
        # Print summary
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        print(f"Passed: {len(passed)}")
        print(f"Failed: {len(failed)}")
        
        if failed:
            print("\nFailed packages:")
            for pkg in failed:
                print(f"  - {pkg}")
            return 1
        
        return 0
        
    finally:
        # Restore original workspace
        if not args.keep_workspace:
            restore_workspace(project_root, original_content)
            print("\nRestored original Nargo.toml")
        else:
            print("\nKept modified Nargo.toml (--keep-workspace)")


if __name__ == "__main__":
    sys.exit(main())
