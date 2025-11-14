# Disaster Prediction Module - Setup Guide

## What's New

A new **Disaster Prediction** page has been integrated into your disaster management app. It uses Google's Gemini AI to analyze images and classify disaster types.

## Features

- Upload disaster images (drag & drop or click to upload)
- AI-powered classification into 9 disaster types:
  - Earthquake
  - Flood
  - Wildfire
  - Hurricane
  - Landslide
  - Drought
  - Tornado
  - Tsunami
  - Volcanic Eruption
- Confidence score and severity level (low/medium/high)
- Detailed analysis description

## Setup Instructions

### 1. Get Gemini API Key

1. Visit: https://makersuite.google.com/app/apikey
2. Sign in with your Google account
3. Create a new API key
4. Copy the key

### 2. Update Environment Variables

Add to your `.env` file:

```
GEMINI_API_KEY=your_gemini_api_key_here
```

### 3. Install Dependencies

The required package `google-generativeai` has been added to `requirements.txt`.

Run:
```bash
pip install -r requirements.txt
```

Or install just the new dependency:
```bash
pip install google-generativeai==0.8.3
```

### 4. Run the App

```bash
python app.py
```

Visit: http://127.0.0.1:5000/disaster-prediction

## Changes Made

### Files Modified:
- `app.py` - Added disaster prediction routes and Gemini integration
- `requirements.txt` - Added google-generativeai dependency
- `templates/base.html` - Added "Disaster AI" to navigation
- `templates/index.html` - Added disaster prediction button
- `README.md` - Updated documentation

### Files Created:
- `templates/disaster_prediction.html` - New disaster prediction page
- `uploads/` folder - Temporary storage for uploaded images (auto-created)

## Usage

1. Navigate to "Disaster AI" in the navigation menu
2. Upload an image of a disaster scene
3. Click "Predict Disaster Type"
4. View the AI analysis with:
   - Disaster type classification
   - Confidence percentage
   - Severity level
   - Detailed description

## Deployment Notes

For Render deployment, make sure to add the `GEMINI_API_KEY` environment variable in your Render dashboard:

1. Go to your service settings
2. Navigate to "Environment" tab
3. Add: `GEMINI_API_KEY` = `your_api_key`
4. Save and redeploy

## API Endpoint

**POST** `/api/predict-disaster`

Request:
- Content-Type: multipart/form-data
- Body: file (image file)

Response:
```json
{
  "disaster_type": "earthquake",
  "confidence": 0.95,
  "description": "The image shows collapsed buildings...",
  "severity": "high"
}
```
