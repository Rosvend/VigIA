# Crime Prediction & Patrol Route Optimization in Medellín

## Overview
This project aims to enhance police patrol efficiency in Medellín by predicting high-risk crime areas and optimizing patrol routes accordingly. Using historical crime data, we train a model to estimate crime likelihood in different grid cells of the city. These predictions are visualized on an interactive map, with color-coded risk levels (e.g., red for high-risk areas).

Based on these predictions, the system generates optimized patrol routes from police stations to the high-risk locations and back, ensuring effective resource allocation.

## Contributors
- **Data Scientist:** Roy Sandoval (@rosvend)  
- **Software Engineers:**  
  - Fernando Gonzales (@fergonzr)
  - Argenis Omaña (@4rg3n15)  

## Features

### 🚨 Crime Prediction & Visualization
- Machine learning models predict crime likelihood for each grid cell in the city.  
- An interactive map displays high-risk areas using color-coded heatmaps.  
- Historical crime trends are analyzed through charts and reports.  

### 🚔 Patrol Route Optimization
- Routes are generated from police stations to high-risk locations and back.  
- Dynamic routing adapts based on risk levels and station jurisdictions.  
- Uses a mapping API to calculate and visualize optimal patrol paths.  

## Target Users
- **Law Enforcement:** Improve patrol efficiency and crime prevention strategies.  
- **City Planners:** Analyze crime distribution for better urban planning.  
- **Government Agencies:** Allocate resources based on data-driven insights.  

## Tech Stack
- **Frontend:** React (JavaScript)  
- **Backend:** Flask (Python)  
- **Database:** PostgreSQL
- **Machine Learning:** Scikit-learn, TensorFlow & PyTorch (Python)  
- **Mapping & Routing:**  
  - OpenStreetMap (for interactive maps)  
  - OpenRouteService **API** (for patrol route optimization)  
- **Visualization:** Leaflet

## Architecture
The project consists of three main components:

### ML Pipeline
- Data ingestion and cleaning
- Spatial grid creation using H3 hexagons
- Feature engineering (spatial, temporal, and categorical)
- Model training and evaluation
- Performance reports with metrics like ROC-AUC
## Backend (Flask)
- RESTful API for crime predictions and route optimization
- PostgreSQL database integration
- API documentation via Swagger/Flasgger
- Error handling and logging
### Frontend (React + Vite)
- Interactive map visualization
- Route planning interface
- Crime heatmap display

## How to run

1. Clone the repository.
2. Get an [OpenRouteService](https://openrouteservice.org/) API key.
3. Make sure docker and docker compose is installed on your machine.
4. Run `ORS_KEY=YOUR_API_KEY docker compose up`.
5. Wait for the images to be built and run.
