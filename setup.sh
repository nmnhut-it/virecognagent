#!/bin/bash
# ViRecognAgent — One-command setup
# Run: bash setup.sh

set -e

echo "════════════════════════════════════════════════"
echo "  ViRecognAgent Setup"
echo "════════════════════════════════════════════════"

# Check Python
python3 --version || { echo "Python 3 required"; exit 1; }

# Create virtual environment
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi
source .venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -q --upgrade pip
pip install -q -r requirements.txt

# Verify key imports
echo ""
echo "Verifying imports..."
python3 -c "
import cv2; print(f'  OpenCV: {cv2.__version__}')
import jiwer; print(f'  jiwer: OK')
import editdistance; print(f'  editdistance: OK')
print('  Core dependencies OK')
"

# Test Vietnamese tools
echo ""
echo "Testing Vietnamese linguistic tools..."
python3 -c "
from src.viet_tools import validate_vietnamese_text, is_valid_syllable_structure
r = validate_vietnamese_text('Xin chào Việt Nam')
print(f'  Syllable validation: {r.valid_syllables}/{r.total_syllables} valid')
print(f'  Confidence: {r.confidence:.2f}')
assert r.confidence > 0.5, 'Syllable validation failed'
print('  Vietnamese tools OK')
"

# Test metrics
echo ""
echo "Testing DER metric..."
python3 -c "
from src.metrics import evaluate
m = evaluate(['Xin chao Viet Nam'], ['Xin chào Việt Nam'])
print(f'  CER: {m.cer:.4f}')
print(f'  DER: {m.der:.4f}')
print(f'  Diacritic chars in ref: {m.n_diacritic_chars}')
assert m.der > 0, 'DER should be > 0 for stripped diacritics'
print('  Metrics OK')
"

echo ""
echo "════════════════════════════════════════════════"
echo "  Setup complete!"
echo ""
echo "  Next steps:"
echo "    1. python scripts/download_datasets.py"
echo "    2. python scripts/run_experiments.py --mode vietocr --max-samples 10"
echo "════════════════════════════════════════════════"
