# Deploying generate_report.py to Google Cloud Run Jobs

This guide walks you through scheduling your daily report script to run automatically on Google Cloud.

## Prerequisites

- Google Cloud account (free tier works)
- Google Cloud CLI installed locally
- Your Polygon.io API key
- SMTP credentials for sending emails

---

## Part 1: Local Setup

### Step 1: Install Google Cloud CLI

1. Download from: https://cloud.google.com/sdk/docs/install
2. Run the installer
3. Open a new terminal and run:
   ```bash
   gcloud init
   ```
4. Follow prompts to log in and select/create a project

### Step 2: Create a Dockerfile

Create this file in your project root (`C:\Users\zmbur\PycharmProjects\backtester\Dockerfile`):

```dockerfile
FROM python:3.11-slim

# Install wkhtmltopdf for PDF generation
RUN apt-get update && apt-get install -y \
    wkhtmltopdf \
    fonts-liberation \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first (for caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Set environment variable for wkhtmltopdf path
ENV WKHTMLTOPDF_PATH=/usr/bin/wkhtmltopdf

# Run the script
CMD ["python", "scripts/generate_report.py"]
```

### Step 3: Update requirements.txt

Make sure your `requirements.txt` includes all dependencies:

```txt
pandas
numpy
plotly
mplfinance
matplotlib
polygon-api-client
pandas-market-calendars
pdfkit
requests
```

### Step 4: Create .gcloudignore

Create `.gcloudignore` in your project root to exclude unnecessary files:

```
.git
.gitignore
__pycache__
*.pyc
venv/
.env
.idea/
data/
charts/
reports/
*.log
```

---

## Part 2: Google Cloud Console Setup

### Step 5: Create a Google Cloud Project

1. Go to https://console.cloud.google.com
2. Click the project dropdown at the top
3. Click "New Project"
4. Name it something like `backtester-reports`
5. Click "Create"
6. Select your new project from the dropdown

### Step 6: Enable Required APIs

Run these commands in your terminal (or enable via Console UI):

```bash
gcloud services enable cloudbuild.googleapis.com
gcloud services enable run.googleapis.com
gcloud services enable cloudscheduler.googleapis.com
gcloud services enable secretmanager.googleapis.com
```

### Step 7: Store Secrets Securely

Based on your codebase, you need these secrets:

```bash
# Polygon API Key (currently hardcoded in polygon_queries.py)
echo -n "b_s_dRysgNN_kZF_nzxwSLdvClTyopGgxtJSqX" | gcloud secrets create POLYGON_API_KEY --data-file=-

# Gmail Password (used in config.py send_email function)
echo -n "YOUR_GMAIL_APP_PASSWORD" | gcloud secrets create GMAIL_PASSWORD --data-file=-
```

**Important**: For Gmail, you need an "App Password" not your regular password:
1. Go to https://myaccount.google.com/apppasswords
2. Generate a new app password for "Mail"
3. Use that 16-character password above

### Step 8: Grant Secret Access to Cloud Run

```bash
# Get your project number
PROJECT_NUMBER=$(gcloud projects describe $(gcloud config get-value project) --format='value(projectNumber)')

# Grant the Cloud Run service account access to secrets
gcloud secrets add-iam-policy-binding POLYGON_API_KEY \
    --member="serviceAccount:${PROJECT_NUMBER}-compute@developer.gserviceaccount.com" \
    --role="roles/secretmanager.secretAccessor"

gcloud secrets add-iam-policy-binding GMAIL_PASSWORD \
    --member="serviceAccount:${PROJECT_NUMBER}-compute@developer.gserviceaccount.com" \
    --role="roles/secretmanager.secretAccessor"
```

---

## Part 3: Build and Deploy

### Step 9: Build the Container Image

From your project directory:

```bash
cd C:\Users\zmbur\PycharmProjects\backtester

# Build and push to Google Container Registry
gcloud builds submit --tag gcr.io/$(gcloud config get-value project)/daily-report
```

This uploads your code and builds the Docker image in the cloud. Takes 3-5 minutes.

### Step 10: Create the Cloud Run Job

```bash
gcloud run jobs create daily-report-job \
    --image gcr.io/$(gcloud config get-value project)/daily-report \
    --region us-east1 \
    --memory 2Gi \
    --cpu 1 \
    --max-retries 1 \
    --task-timeout 15m \
    --set-secrets="POLYGON_API_KEY=POLYGON_API_KEY:latest,GMAIL_PASSWORD=GMAIL_PASSWORD:latest"
```

**Note**: Adjust `--region` to match your timezone preference:
- `us-east1` = Virginia (Eastern Time)
- `us-central1` = Iowa
- `us-west1` = Oregon (Pacific Time)

### Step 11: Test the Job Manually

```bash
gcloud run jobs execute daily-report-job --region us-east1
```

Check the logs:
```bash
gcloud run jobs executions list --job daily-report-job --region us-east1
gcloud logging read "resource.type=cloud_run_job" --limit 50
```

---

## Part 4: Schedule the Job

### Step 12: Create a Cloud Scheduler Job

Schedule it to run at 6:30 AM Eastern Time on weekdays:

```bash
# Create a service account for the scheduler
gcloud iam service-accounts create scheduler-sa --display-name="Cloud Scheduler Service Account"

# Grant it permission to invoke Cloud Run jobs
gcloud run jobs add-iam-policy-binding daily-report-job \
    --region us-east1 \
    --member="serviceAccount:scheduler-sa@$(gcloud config get-value project).iam.gserviceaccount.com" \
    --role="roles/run.invoker"

# Create the scheduled job
gcloud scheduler jobs create http daily-report-schedule \
    --location us-east1 \
    --schedule "30 6 * * 1-5" \
    --time-zone "America/New_York" \
    --uri "https://us-east1-run.googleapis.com/apis/run.googleapis.com/v1/namespaces/$(gcloud config get-value project)/jobs/daily-report-job:run" \
    --http-method POST \
    --oauth-service-account-email "scheduler-sa@$(gcloud config get-value project).iam.gserviceaccount.com"
```

**Cron schedule explained**: `30 6 * * 1-5`
- `30` = minute 30
- `6` = 6 AM
- `*` = every day of month
- `*` = every month
- `1-5` = Monday through Friday

### Step 13: Test the Scheduler

```bash
gcloud scheduler jobs run daily-report-schedule --location us-east1
```

---

## Part 5: Update Your Code for Cloud

### Step 14: Modify polygon_queries.py for Secrets

Update `data_queries/polygon_queries.py` line 11 to read from environment variable:

```python
# Change this line:
# poly_client = RESTClient(api_key="b_s_dRysgNN_kZF_nzxwSLdvClTyopGgxtJSqX")

# To this:
import os
poly_client = RESTClient(api_key=os.environ.get("POLYGON_API_KEY", "b_s_dRysgNN_kZF_nzxwSLdvClTyopGgxtJSqX"))
```

Your `config.py` already reads `GMAIL_PASSWORD` from environment variables via `os.getenv()`, so no changes needed there.

### Step 15: Handle the charts/reports directories

Since Cloud Run is ephemeral, modify `generate_report.py` to use `/tmp` for temporary files:

```python
import tempfile

# Replace hardcoded paths with temp directories
charts_dir = tempfile.mkdtemp(prefix="charts_")
reports_dir = tempfile.mkdtemp(prefix="reports_")
```

Or simply remove local PDF saving since you're emailing the report anyway.

---

## Part 6: Ongoing Maintenance

### Updating the Code

After making changes, redeploy:

```bash
# Rebuild the image
gcloud builds submit --tag gcr.io/$(gcloud config get-value project)/daily-report

# Update the job to use the new image
gcloud run jobs update daily-report-job \
    --image gcr.io/$(gcloud config get-value project)/daily-report \
    --region us-east1
```

### Viewing Logs

```bash
# Recent executions
gcloud run jobs executions list --job daily-report-job --region us-east1

# Detailed logs
gcloud logging read "resource.type=cloud_run_job AND resource.labels.job_name=daily-report-job" --limit 100

# Or use the Console UI:
# https://console.cloud.google.com/run/jobs
```

### Pausing the Schedule

```bash
gcloud scheduler jobs pause daily-report-schedule --location us-east1
```

### Resuming the Schedule

```bash
gcloud scheduler jobs resume daily-report-schedule --location us-east1
```

### Changing the Schedule

```bash
gcloud scheduler jobs update http daily-report-schedule \
    --location us-east1 \
    --schedule "0 7 * * 1-5"  # Change to 7:00 AM
```

---

## Cost Estimate

For a single daily job running ~5 minutes:
- **Cloud Run**: ~$0.00 (free tier: 2 million requests/month)
- **Cloud Build**: ~$0.00 (free tier: 120 build-minutes/day)
- **Cloud Scheduler**: ~$0.10/month (3 free jobs, then $0.10/job/month)
- **Secret Manager**: ~$0.00 (free tier covers this)

**Total: Effectively free** for your use case.

---

## Troubleshooting

### Job fails immediately
- Check logs: `gcloud logging read "resource.type=cloud_run_job" --limit 50`
- Common issues: missing dependencies, incorrect secret names

### Scheduler doesn't trigger
- Verify the service account has `roles/run.invoker` permission
- Check scheduler logs in Console UI

### PDF generation fails
- Ensure wkhtmltopdf is installed in the Dockerfile
- Check the `WKHTMLTOPDF_PATH` environment variable

### Import errors
- Make sure all dependencies are in `requirements.txt`
- Check that file paths work in Linux (case-sensitive!)

### Timeout errors
- Increase `--task-timeout` when creating the job
- Maximum is 1 hour for Cloud Run Jobs

---

## Quick Reference Commands

```bash
# Test run
gcloud run jobs execute daily-report-job --region us-east1

# View recent executions
gcloud run jobs executions list --job daily-report-job --region us-east1

# View logs
gcloud logging read "resource.type=cloud_run_job" --limit 50

# Rebuild and redeploy
gcloud builds submit --tag gcr.io/$(gcloud config get-value project)/daily-report && \
gcloud run jobs update daily-report-job --image gcr.io/$(gcloud config get-value project)/daily-report --region us-east1

# Pause/resume scheduler
gcloud scheduler jobs pause daily-report-schedule --location us-east1
gcloud scheduler jobs resume daily-report-schedule --location us-east1
```
