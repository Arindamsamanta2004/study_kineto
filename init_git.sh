#!/bin/bash

# Git Repository Initialization Script
# Prepares the project for GitHub upload

set -e

echo "=========================================="
echo "  Git Repository Initialization"
echo "=========================================="
echo ""

# Check if git is installed
if ! command -v git &> /dev/null; then
    echo "ERROR: git is not installed"
    exit 1
fi

# Initialize git repository if not already initialized
if [ ! -d ".git" ]; then
    echo "Initializing git repository..."
    git init
    echo "✓ Repository initialized"
else
    echo "✓ Git repository already exists"
fi

# Create .git/info/exclude for additional ignores
mkdir -p .git/info
cat > .git/info/exclude << 'EOF'
# Additional local excludes (not in .gitignore)
*.swp
*~
.DS_Store
EOF

# Add all files
echo ""
echo "Adding files to git..."
git add .

# Show status
echo ""
echo "Repository status:"
git status

# Check for commits
if git rev-parse HEAD >/dev/null 2>&1; then
    echo ""
    echo "✓ Repository has existing commits"
    echo ""
    echo "Recent commits:"
    git log --oneline -5
else
    echo ""
    echo "Creating initial commit..."
    git commit -m "Initial commit: Kineto vs NSight Compute profiling benchmark

- Vector addition benchmark for GPU profiling
- Kineto profiler integration (PyTorch native)
- NVIDIA NSight Compute profiler integration
- Two configurations: Config A (32x4Kx256) and Config B (64x8Kx512)
- Automated profiling scripts
- Comparison analysis tools
- Comprehensive documentation

Ready to run on H200 for comparative analysis."

    echo "✓ Initial commit created"
fi

echo ""
echo "=========================================="
echo "  Next Steps"
echo "=========================================="
echo ""
echo "1. Create GitHub repository:"
echo "   - Go to https://github.com/new"
echo "   - Name: kineto-nsight-comparison (or your choice)"
echo "   - Description: Comparative analysis of Kineto vs NVIDIA NSight Compute profiling"
echo "   - Public or Private: Your choice"
echo "   - DO NOT initialize with README (we already have one)"
echo ""
echo "2. Add remote and push:"
echo "   git remote add origin https://github.com/YOUR_USERNAME/REPO_NAME.git"
echo "   git branch -M main"
echo "   git push -u origin main"
echo ""
echo "3. On H200 container:"
echo "   git clone https://github.com/YOUR_USERNAME/REPO_NAME.git"
echo "   cd REPO_NAME"
echo "   pip install -r requirements.txt"
echo "   ./run_all_profilers.sh"
echo ""
echo "=========================================="
echo ""

# Show files to be committed
echo "Files ready for GitHub:"
git ls-files | head -20
echo "... (run 'git ls-files' to see all)"
echo ""
echo "✓ Repository is ready for GitHub!"
