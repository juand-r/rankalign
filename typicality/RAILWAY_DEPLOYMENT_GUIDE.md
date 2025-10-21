# Railway Deployment Guide for Typicality Dashboard

## ✅ Pre-deployment Checklist

Your dashboard is ready to deploy! Here's what's been set up:

1. **requirements.txt** - Updated with all dependencies including scipy
2. **Procfile** - Configured to run with gunicorn
3. **railway.toml** - Railway configuration file
4. **.railwayignore** - Excludes unnecessary files from deployment

## 📋 Required Data Files

Your dashboard needs these CSV files (✅ = verified present):
- ✅ `merged_data_google-gemma-2-2b_random_test.csv` 
- ✅ `hypernym_predictions/cars_combined_clean_predictions_with_validator.csv`

## 🚀 Deployment Steps

### Option 1: Deploy via Railway CLI

1. **Install Railway CLI** (if not already installed):
   ```bash
   npm i -g @railway/cli
   ```

2. **Login to Railway**:
   ```bash
   railway login
   ```

3. **Navigate to the typicality directory**:
   ```bash
   cd /datastor1/jdr/gv-gap/rankalign/typicality
   ```

4. **Initialize Railway project** (first time only):
   ```bash
   railway init
   ```

5. **Deploy**:
   ```bash
   railway up
   ```

6. **Get the deployment URL**:
   ```bash
   railway domain
   ```

### Option 2: Deploy via GitHub

1. **Push your code to GitHub** (if not already):
   ```bash
   git add .
   git commit -m "Prepare dashboard for Railway deployment"
   git push origin main
   ```

2. **Connect to Railway**:
   - Go to [railway.app](https://railway.app/)
   - Click "New Project"
   - Choose "Deploy from GitHub repo"
   - Select your repository
   - Railway will automatically detect your configuration

3. **Set the root directory** (if deploying from a subdirectory):
   - In Railway project settings
   - Set "Root Directory" to `typicality`

### Option 3: Deploy from Current Directory

1. **Go to Railway dashboard**: https://railway.app/dashboard
2. **Create a new project**: Click "New Project" → "Empty Project"
3. **Add a service**: Click "Create" → "Empty Service"
4. **Deploy from local directory**:
   ```bash
   cd /datastor1/jdr/gv-gap/rankalign/typicality
   railway link  # Link to your project
   railway up    # Upload and deploy
   ```

## ⚙️ Configuration

### Environment Variables
The dashboard automatically uses Railway's `PORT` environment variable (no manual configuration needed).

### Custom Domain (Optional)
After deployment:
1. Go to your Railway project dashboard
2. Click on your service
3. Go to "Settings" → "Domains"
4. Generate a Railway domain or add your custom domain

## 🔍 Verify Deployment

After deployment, your dashboard will be available at the Railway-provided URL. The app should:
- Load the typicality analysis scatter plot
- Show the cars hypernym predictions scatter plot
- Allow you to change color schemes via the dropdown

## 🐛 Troubleshooting

### Build Fails
- Check Railway logs: `railway logs`
- Verify all dependencies are in requirements.txt

### Data Files Not Found
Make sure the CSV files are committed to your repository:
```bash
git add merged_data_google-gemma-2-2b_random_test.csv
git add hypernym_predictions/cars_combined_clean_predictions_with_validator.csv
git commit -m "Add data files for dashboard"
```

### App Crashes on Startup
- Check logs for missing dependencies
- Verify the data files are in the correct location
- Ensure `dashboard.py` is in the root of your deployment directory

## 📊 Local Testing

Before deploying, test locally:
```bash
cd /datastor1/jdr/gv-gap/rankalign/typicality
pip install -r requirements.txt
python dashboard.py
# Visit http://localhost:8050
```

## 💡 Tips

1. **Deployment Size**: The .railwayignore file excludes log files and cached data to keep deployment lean
2. **Updates**: After making changes, simply run `railway up` again to redeploy
3. **Logs**: Monitor logs with `railway logs` or view them in the Railway dashboard
4. **Performance**: The app subsamples the cars data for better performance (10% sample)

## 🎯 Quick Start (Fastest Method)

```bash
cd /datastor1/jdr/gv-gap/rankalign/typicality
railway login
railway init
railway up
railway open  # Opens your deployed app in browser
```

---

**Need help?** Check Railway docs at https://docs.railway.app/

