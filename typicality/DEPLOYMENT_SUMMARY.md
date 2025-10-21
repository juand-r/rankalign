# Dashboard Deployment Summary

## ✅ Files Created

All files are in the `typicality/` directory:

1. **`dashboard.py`** - Main interactive dashboard with Plotly Dash
   - Generator vs Validator scatter (with hover showing noun pairs)
   - Typicality comparison plots
   - G-V gap analysis
   - Model performance statistics

2. **`requirements.txt`** - Python dependencies
   - dash, plotly, pandas, numpy, gunicorn

3. **`Procfile`** - Railway deployment command

4. **`railway.toml`** - Railway configuration

5. **`DASHBOARD_README.md`** - Comprehensive documentation

## 🚀 Quick Start

### Local Testing:
```bash
cd /datastor1/jdr/gv-gap/rankalign/typicality
source ~/venvs/venv_lexcons/bin/activate
pip install -r requirements.txt
python dashboard.py
```
Open http://localhost:8050

### Deploy to Railway:

**Method 1: Railway CLI**
```bash
cd typicality/
railway login
railway init
railway up
```

**Method 2: Railway Web UI**
1. Go to https://railway.app
2. New Project → Deploy from GitHub repo
3. Set root directory to `typicality/`
4. Deploy!

## 📊 Dashboard Features

### Main Scatter Plot (Left)
- **X-axis**: Generator score (as you requested!)
- **Y-axis**: Validator score
- **Colors**: Red = negative examples, Blue = positive examples
- **Hover**: Shows (noun1, noun2) pair for each point
- **Reference**: y=x line showing perfect agreement

### Typicality Panel (Right)
- Dropdown to select typicality measure:
  - P(noun2) - Unconditional hypernym
  - P(noun1) - Unconditional hyponym
  - P(noun2|context) - Conditional hypernym
- Shows correlation coefficient
- Hover displays noun pairs

### G-V Gap Analysis (Bottom)
- Statistics cards for overall, positive, and negative examples
- Interactive plot showing G-V gap vs typicality
- Model performance comparison (baseline vs with typicality)

## 📁 Project Structure

```
typicality/
├── dashboard.py                           # ⭐ Main app
├── requirements.txt                       # Dependencies
├── Procfile                               # Railway config
├── railway.toml                           # Railway config
├── DASHBOARD_README.md                    # Full docs
├── merged_data_google-gemma-2-2b_random_seed0.csv  # Data (required)
│
├── compute_gpt2_typicality.py            # Scripts
├── merge_data.py
├── eda_typicality.py
├── test_additive_model.py
├── residual_analysis.py
│
└── eda_outputs_*/                         # Analysis outputs
```

## ✨ What Makes It Special

1. **Interactive**: Hover over any point to see the exact noun pair
2. **Organized**: All code in typicality/ directory
3. **Railway-ready**: One command deployment
4. **Informative**: Shows correlations, statistics, and model fits
5. **Professional**: Clean UI with proper styling
6. **Responsive**: Auto-adjusts to screen size

## 🔧 Tested and Working

- ✅ Dashboard loads successfully
- ✅ Finds and loads CSV data automatically
- ✅ All dependencies specified
- ✅ Railway configuration complete
- ✅ Generator on x-axis as requested
- ✅ Hover shows (noun1, noun2) pairs

## 📝 Next Steps

1. Test locally: `python dashboard.py`
2. Push to GitHub (if using Railway GitHub integration)
3. Deploy to Railway
4. Share the URL!

The dashboard will be live at: `https://your-project-name.up.railway.app`

