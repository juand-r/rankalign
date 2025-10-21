# Typicality Analysis Dashboard

Interactive dashboard for exploring the relationship between generator-validator gap and typicality scores.

## Features

- **Interactive Scatter Plots**: Hover over points to see (noun1, noun2) pairs
- **Generator vs Validator**: Main plot showing the G-V gap with color coding for positive/negative examples
- **Typicality Analysis**: Switchable views for different typicality measures
- **G-V Gap Visualization**: Shows how typicality correlates with the generator-validator gap
- **Model Performance Stats**: Real-time computation of regression statistics

## Local Development

### Install dependencies:
```bash
pip install -r requirements.txt
```

### Run locally:
```bash
python dashboard.py
```

Then open http://localhost:8050 in your browser.

## Deploy on Railway

### Option 1: Railway CLI

1. Install Railway CLI:
   ```bash
   npm install -g @railway/cli
   ```

2. Login to Railway:
   ```bash
   railway login
   ```

3. Initialize and deploy:
   ```bash
   cd /path/to/typicality/
   railway init
   railway up
   ```

### Option 2: Railway Web UI

1. Go to https://railway.app
2. Click "New Project" → "Deploy from GitHub repo"
3. Select your repository
4. Set root directory to `typicality/`
5. Railway will auto-detect the configuration
6. Click "Deploy"

### Important Notes

- Make sure `merged_data_*seed0.csv` is present in the typicality/ directory
- The dashboard automatically finds and loads the CSV file
- Railway will use the `Procfile` and `railway.toml` for configuration
- Environment variable `PORT` is automatically set by Railway

## File Structure

```
typicality/
├── dashboard.py                    # Main dashboard application
├── requirements.txt                # Python dependencies
├── Procfile                        # Railway start command
├── railway.toml                    # Railway configuration
├── merged_data_*seed0.csv         # Data file (required)
├── compute_gpt2_typicality.py     # Data generation script
├── merge_data.py                  # Data merging script
├── eda_typicality.py              # EDA script
├── test_additive_model.py         # Model testing script
└── eda_outputs_*/                 # Analysis outputs
```

## Dashboard Layout

### Top Panel
- **Left**: Generator vs Validator scatter plot (hover to see noun pairs)
- **Right**: Typicality vs Generator with dropdown to select typicality measure

### Middle Panel
- Statistics cards showing G-V gap for overall, positive, and negative examples
- G-V Gap vs Typicality plot

### Bottom Panel
- Model performance statistics showing baseline and typicality-enhanced models

## Data Requirements

The dashboard requires a CSV file named `merged_data_*seed0.csv` with columns:
- `noun1`, `noun2`: Word pairs
- `ground_truth`: Binary label (0/1)
- `gen_score`: Generator log probability
- `disc_score`: Discriminator/validator score
- `log_prob_noun2`: Unconditional hypernym probability from GPT-2
- `log_prob_noun1`: Unconditional hyponym probability from GPT-2
- `log_prob_noun2_given_context`: Conditional hypernym probability from GPT-2

## Troubleshooting

### "No merged data file found"
Make sure you have a `merged_data_*seed0.csv` file in the typicality/ directory.

### Railway deployment fails
Check that:
1. All required files (dashboard.py, requirements.txt, Procfile) are present
2. The CSV data file is included
3. Root directory is set to `typicality/` in Railway settings

### Port issues
Railway automatically sets the PORT environment variable. The app is configured to use it.

