# Orca Deployment Script for Google Cloud Run
# Author: Orca Assistant

$ErrorActionPreference = "Stop"

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "      ORCA DEPLOYMENT SCRIPT" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan

# 1. Check gcloud
Write-Host "`n[1/6] Checking gcloud..." -ForegroundColor Yellow
try {
    gcloud --version | Out-Null
    Write-Host "✓ gcloud is installed." -ForegroundColor Green
} catch {
    Write-Error "gcloud CLI is not installed or not in PATH. Please install Google Cloud SDK."
}

# 2. Get Project ID
$currentProject = gcloud config get-value project 2>$null
if ([string]::IsNullOrWhiteSpace($currentProject)) {
    $currentProject = "YOUR_PROJECT_ID"
}

Write-Host "`nCurrent Project ID: $currentProject"
$projectID = Read-Host "Enter your Google Cloud Project ID (Press Enter to use '$currentProject')"
if ([string]::IsNullOrWhiteSpace($projectID)) {
    $projectID = $currentProject
}

Write-Host "Using Project ID: $projectID" -ForegroundColor Cyan
gcloud config set project $projectID

# 3. Get Region
$defaultRegion = "us-central1"
$region = Read-Host "Enter Region (Press Enter to use '$defaultRegion')"
if ([string]::IsNullOrWhiteSpace($region)) {
    $region = $defaultRegion
}
Write-Host "Using Region: $region" -ForegroundColor Cyan

# 4. Enable APIs
Write-Host "`n[2/6] Enabling required APIs (this may take a minute)..." -ForegroundColor Yellow
gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com
Write-Host "✓ APIs enabled." -ForegroundColor Green

# 5. Build Container
Write-Host "`n[3/6] Building container image..." -ForegroundColor Yellow
$imageName = "gcr.io/$projectID/orca-web"
Write-Host "Image: $imageName"

# Submit build to Cloud Build
gcloud builds submit --tag $imageName .

if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ Build successful." -ForegroundColor Green
} else {
    Write-Error "Build failed. Please check the logs above."
}

# 6. Deploy to Cloud Run
Write-Host "`n[4/6] Deploying to Cloud Run..." -ForegroundColor Yellow
$serviceName = "orca-web"

# Deploy command
gcloud run deploy $serviceName `
    --image $imageName `
    --platform managed `
    --region $region `
    --allow-unauthenticated `
    --memory 2Gi `
    --cpu 2 `
    --timeout 300

if ($LASTEXITCODE -eq 0) {
    Write-Host "`n[5/6] Deployment successful!" -ForegroundColor Green
    
    # Get URL
    $url = gcloud run services describe $serviceName --platform managed --region $region --format 'value(status.url)'
    
    Write-Host "`n==========================================" -ForegroundColor Cyan
    Write-Host "   ORCA IS LIVE!" -ForegroundColor Cyan
    Write-Host "   URL: $url" -ForegroundColor Green
    Write-Host "==========================================" -ForegroundColor Cyan
    
    # Open URL
    Start-Process $url
} else {
    Write-Error "Deployment failed. Please check the logs above."
}
