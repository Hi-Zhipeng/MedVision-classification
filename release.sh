#!/bin/bash
# PyPI Release Script for MedVision-Classification

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_step() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

# Function to show usage
show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -h, --help         Show this help message"
    echo "  --test             Upload to TestPyPI instead of PyPI"
    echo "  --dry-run          Build package but don't upload"
    echo "  --check-only       Only check package without building"
    echo ""
    echo "Examples:"
    echo "  $0                 # Build and upload to PyPI"
    echo "  $0 --test          # Build and upload to TestPyPI"
    echo "  $0 --dry-run       # Build package only"
    echo "  $0 --check-only    # Check package configuration only"
}

# Parse command line arguments
TEST_PYPI=false
DRY_RUN=false
CHECK_ONLY=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_usage
            exit 0
            ;;
        --test)
            TEST_PYPI=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --check-only)
            CHECK_ONLY=true
            shift
            ;;
        --*)
            print_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
        *)
            print_error "Unknown argument: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Check if required tools are installed
print_step "Checking required tools..."

if ! command -v python &> /dev/null; then
    print_error "Python is not installed or not in PATH"
    exit 1
fi

if ! python -c "import build" &> /dev/null; then
    print_warning "build package not found, installing..."
    pip install build
fi

if ! python -c "import twine" &> /dev/null; then
    print_warning "twine package not found, installing..."
    pip install twine
fi

print_status "All required tools are available"

# Check package configuration
print_step "Checking package configuration..."

# Verify pyproject.toml exists and is valid
if [[ ! -f "pyproject.toml" ]]; then
    print_error "pyproject.toml not found"
    exit 1
fi

# Check for required files
REQUIRED_FILES=("README.md" "LICENSE")
for file in "${REQUIRED_FILES[@]}"; do
    if [[ ! -f "$file" ]]; then
        print_error "Required file not found: $file"
        exit 1
    fi
done

# Validate package structure
if [[ ! -d "medvision_cls" ]]; then
    print_error "Package directory 'medvision_cls' not found"
    exit 1
fi

print_status "Package configuration is valid"

# Check if we should only do checks
if [[ "$CHECK_ONLY" == true ]]; then
    print_status "Check completed successfully"
    exit 0
fi

# Clean previous builds
print_step "Cleaning previous builds..."
rm -rf build/ dist/ *.egg-info/
print_status "Cleaned previous builds"

# Build the package
print_step "Building package..."
python -m build
print_status "Package built successfully"

# Check the built package
print_step "Checking built package..."
python -m twine check dist/*
print_status "Package check passed"

# Show package contents
print_step "Package contents:"
ls -la dist/

# If dry run, stop here
if [[ "$DRY_RUN" == true ]]; then
    print_status "Dry run completed. Package is ready but not uploaded."
    exit 0
fi

# Determine upload target
if [[ "$TEST_PYPI" == true ]]; then
    UPLOAD_TARGET="testpypi"
    UPLOAD_URL="https://test.pypi.org/legacy/"
    print_warning "Uploading to TestPyPI"
else
    UPLOAD_TARGET="pypi"
    UPLOAD_URL="https://upload.pypi.org/legacy/"
    print_warning "Uploading to PyPI"
fi

# Confirm upload
echo ""
print_step "Package summary:"
python -c "
import toml
config = toml.load('pyproject.toml')
project = config['project']
print(f\"  Name: {project['name']}\")
print(f\"  Version: {project['version']}\")
print(f\"  Description: {project['description']}\")
print(f\"  Author: {project['authors'][0]['name']} <{project['authors'][0]['email']}>\")
"

echo ""
read -p "Do you want to upload to $UPLOAD_TARGET? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    print_status "Upload cancelled"
    exit 0
fi

# Upload the package
print_step "Uploading package to $UPLOAD_TARGET..."

if [[ "$TEST_PYPI" == true ]]; then
    python -m twine upload --repository testpypi dist/*
else
    python -m twine upload dist/*
fi

print_status "Package uploaded successfully!"

# Show installation instructions
echo ""
print_step "Installation instructions:"
if [[ "$TEST_PYPI" == true ]]; then
    echo "  pip install --index-url https://test.pypi.org/simple/ medvision-classification"
else
    echo "  pip install medvision-classification"
fi

print_status "Release completed successfully!"
