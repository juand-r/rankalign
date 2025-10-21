#!/bin/bash
# Quick test script for the dashboard

echo "=========================================="
echo "Testing Typicality Dashboard"
echo "=========================================="

# Check if data file exists
if ls merged_data_*seed0.csv 1> /dev/null 2>&1; then
    echo "✓ Data file found"
else
    echo "✗ No data file found (need merged_data_*seed0.csv)"
    exit 1
fi

# Check Python dependencies
echo ""
echo "Checking dependencies..."
source ~/venvs/venv_lexcons/bin/activate

python -c "import dash; print('✓ dash installed')" || { echo "✗ dash not installed"; exit 1; }
python -c "import plotly; print('✓ plotly installed')" || { echo "✗ plotly not installed"; exit 1; }
python -c "import pandas; print('✓ pandas installed')" || { echo "✗ pandas not installed"; exit 1; }

# Test dashboard loads
echo ""
echo "Testing dashboard loads..."
python -c "from dashboard import app, df; print(f'✓ Dashboard loaded successfully')" || { echo "✗ Dashboard failed to load"; exit 1; }
python -c "from dashboard import df; print(f'✓ Data loaded: {len(df)} examples')"

echo ""
echo "=========================================="
echo "✓ All checks passed!"
echo "=========================================="
echo ""
echo "To start the dashboard:"
echo "  python dashboard.py"
echo ""
echo "Then open: http://localhost:8050"
echo ""
echo "To deploy on Railway:"
echo "  railway login"
echo "  railway init"
echo "  railway up"
echo "=========================================="

