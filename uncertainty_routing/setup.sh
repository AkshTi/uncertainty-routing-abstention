#!/bin/bash
# Save as: setup.sh
# Run with: bash setup.sh

echo "=========================================="
echo " Setting up Uncertainty Expression Gate"
echo " ICLR Workshop Experiments"
echo "=========================================="
echo ""

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "❌ Conda not found. Please install Anaconda or Miniconda first."
    echo "   Download from: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

echo "✓ Conda found"
echo ""

# Create environment
echo "Creating conda environment 'mech_interp'..."
conda create -n mech_interp python=3.10 -y

if [ $? -ne 0 ]; then
    echo "❌ Failed to create conda environment"
    exit 1
fi

echo "✓ Environment created"
echo ""

# Activate environment
echo "Activating environment..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate mech_interp

if [ $? -ne 0 ]; then
    echo "❌ Failed to activate environment"
    exit 1
fi

echo "✓ Environment activated"
echo ""

# Install PyTorch (with CUDA if available)
echo "Installing PyTorch..."
if command -v nvidia-smi &> /dev/null; then
    echo "  GPU detected, installing PyTorch with CUDA..."
    conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
else
    echo "  No GPU detected, installing CPU-only PyTorch..."
    conda install pytorch torchvision torchaudio cpuonly -c pytorch -y
fi

if [ $? -ne 0 ]; then
    echo "❌ Failed to install PyTorch"
    exit 1
fi

echo "✓ PyTorch installed"
echo ""

# Install other requirements
echo "Installing other requirements..."
pip install transformers>=4.35.0 \
            numpy>=1.24.0 \
            pandas>=2.0.0 \
            matplotlib>=3.7.0 \
            seaborn>=0.12.0 \
            scikit-learn>=1.3.0 \
            tqdm>=4.65.0 \
            huggingface-hub>=0.19.0

if [ $? -ne 0 ]; then
    echo "❌ Failed to install requirements"
    exit 1
fi

echo "✓ All requirements installed"
echo ""

# Create directory structure
echo "Creating directory structure..."
mkdir -p data
mkdir -p results

echo "✓ Directories created"
echo ""

# Prepare datasets
echo "Preparing datasets..."
python data_preparation.py

if [ $? -ne 0 ]; then
    echo "❌ Failed to prepare datasets"
    exit 1
fi

echo "✓ Datasets prepared"
echo ""

# Run tests
echo "Running test suite..."
python test_everything.py

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Some tests failed. Please check the output above."
    exit 1
fi

echo ""
echo "=========================================="
echo " ✓ Setup Complete!"
echo "=========================================="
echo ""
echo "Your environment is ready to use."
echo ""
echo "Next steps:"
echo "  1. Activate environment:  conda activate mech_interp"
echo "  2. Quick test:           python experiment5_trustworthiness.py"
echo "  3. Run all experiments:  See README.md"
echo ""
