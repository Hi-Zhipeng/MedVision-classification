#!/bin/bash
# Pre-release check script

echo "🔍 Running pre-release checks for MedVision-Classification..."

# Check if required files exist
echo "📁 Checking required files..."
required_files=("pyproject.toml" "README.md" "LICENSE" "medvision_cls/__init__.py")
for file in "${required_files[@]}"; do
    if [[ -f "$file" ]]; then
        echo "  ✅ $file"
    else
        echo "  ❌ $file (missing)"
        exit 1
    fi
done

# Check package structure
echo "📦 Checking package structure..."
if [[ -d "medvision_cls" ]]; then
    echo "  ✅ medvision_cls package directory"
else
    echo "  ❌ medvision_cls package directory (missing)"
    exit 1
fi

# Check if all submodules have __init__.py
echo "🐍 Checking Python package structure..."
for dir in medvision_cls/*/; do
    if [[ -d "$dir" && ! -f "${dir}__init__.py" ]]; then
        echo "  ⚠️  Missing __init__.py in $dir"
    fi
done

# Test import
echo "🧪 Testing package import..."
if python -c "import medvision_cls; print(f'Version: {medvision_cls.__version__}')" 2>/dev/null; then
    echo "  ✅ Package import successful"
else
    echo "  ❌ Package import failed"
    exit 1
fi

# Check dependencies
echo "📋 Checking dependencies..."
if python -c "
import sys
required = ['torch', 'pytorch_lightning', 'torchmetrics', 'numpy', 'click']
missing = []
for pkg in required:
    try:
        __import__(pkg)
    except ImportError:
        missing.append(pkg)
if missing:
    print(f'Missing packages: {missing}')
    sys.exit(1)
else:
    print('All required dependencies available')
"; then
    echo "  ✅ Dependencies check passed"
else
    echo "  ❌ Dependencies check failed"
    exit 1
fi

# Test CLI
echo "🖥️  Testing CLI..."
if python -m medvision_cls.cli.main --help >/dev/null 2>&1; then
    echo "  ✅ CLI working"
else
    echo "  ❌ CLI not working"
    exit 1
fi

echo ""
echo "🎉 All pre-release checks passed!"
echo ""
echo "📋 Next steps:"
echo "  1. Update version in pyproject.toml if needed"
echo "  2. Update CHANGELOG.md"
echo "  3. Run: ./release.sh --test (for TestPyPI)"
echo "  4. Run: ./release.sh (for PyPI)"
