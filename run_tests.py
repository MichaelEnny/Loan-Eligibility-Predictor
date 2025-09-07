"""
Test runner script for the Feature Engineering Pipeline.
Runs comprehensive tests with coverage reporting.
"""

import subprocess
import sys
from pathlib import Path


def run_tests():
    """Run all tests with coverage reporting."""
    print("🧪 Running Feature Engineering Pipeline Tests")
    print("=" * 60)
    
    # Install test requirements if needed
    try:
        import pytest
        import pytest_cov
    except ImportError:
        print("Installing test dependencies...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", 
            "pytest>=7.0.0", "pytest-cov>=4.0.0", "pytest-xdist>=3.0.0"
        ])
    
    # Run tests with coverage
    test_args = [
        "-m", "pytest",
        "tests/",
        "--cov=feature_engineering",
        "--cov-report=term-missing",
        "--cov-report=html:coverage_html",
        "--cov-fail-under=90",  # Require 90% coverage
        "-v",
        "--tb=short",
        "--strict-markers",
        "-x"  # Stop on first failure for faster debugging
    ]
    
    try:
        result = subprocess.run([sys.executable] + test_args, check=True)
        print("\n✅ All tests passed!")
        print("📊 Coverage report generated in coverage_html/")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Tests failed with exit code {e.returncode}")
        return False


def run_quick_tests():
    """Run tests without coverage for quick validation."""
    print("⚡ Running Quick Tests")
    print("=" * 30)
    
    test_args = [
        "-m", "pytest",
        "tests/",
        "-v",
        "--tb=short",
        "-x"
    ]
    
    try:
        subprocess.run([sys.executable] + test_args, check=True)
        print("\n✅ Quick tests passed!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Tests failed with exit code {e.returncode}")
        return False


def run_specific_test(test_path):
    """Run a specific test file or test function."""
    print(f"🎯 Running specific test: {test_path}")
    print("=" * 40)
    
    test_args = [
        "-m", "pytest",
        test_path,
        "-v",
        "--tb=long"
    ]
    
    try:
        subprocess.run([sys.executable] + test_args, check=True)
        print(f"\n✅ Test {test_path} passed!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Test failed with exit code {e.returncode}")
        return False


def check_test_structure():
    """Check test file structure and completeness."""
    print("🔍 Checking test structure...")
    
    test_dir = Path("tests")
    required_files = [
        "__init__.py",
        "conftest.py",
        "test_config.py",
        "test_encoders.py", 
        "test_preprocessors.py",
        "test_interactions.py",
        "test_dimensionality.py",
        "test_pipeline.py"
    ]
    
    missing_files = []
    for file_name in required_files:
        file_path = test_dir / file_name
        if not file_path.exists():
            missing_files.append(file_name)
        else:
            print(f"  ✓ {file_name}")
    
    if missing_files:
        print(f"\n❌ Missing test files: {missing_files}")
        return False
    else:
        print("\n✅ All required test files present!")
        return True


def main():
    """Main test runner function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Feature Engineering Pipeline Test Runner")
    parser.add_argument(
        "--mode", 
        choices=["full", "quick", "check"],
        default="full",
        help="Test mode to run"
    )
    parser.add_argument(
        "--test", 
        help="Specific test to run (e.g., tests/test_pipeline.py::TestClass::test_method)"
    )
    
    args = parser.parse_args()
    
    # Check test structure first
    if not check_test_structure():
        sys.exit(1)
    
    success = True
    
    if args.test:
        success = run_specific_test(args.test)
    elif args.mode == "full":
        success = run_tests()
    elif args.mode == "quick":
        success = run_quick_tests()
    elif args.mode == "check":
        print("✅ Test structure check completed!")
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()