# Crime Prediction & Patrol Route Optimization in Medellín

## Overview
This project aims to enhance police patrol efficiency in Medellín by predicting high-risk crime areas and optimizing patrol routes accordingly. Using historical crime data, we train a model to estimate crime likelihood in different grid cells of the city. These predictions are visualized on an interactive map, with color-coded risk levels (e.g., red for high-risk areas).

Based on these predictions, the system generates optimized patrol routes from police stations to the high-risk locations and back, ensuring effective resource allocation.

## Team
Data Scientist: Roy Sandoval (@rosvend)
Software Engineers: Fernando Gonzales (@fergonzr), Argenis Omaña (@4rg3n15), Valentina Sánchez (@valentinaSV1028)

## Features

### Crime Prediction & Visualization
Machine learning models predict crime likelihood for each grid cell in the city.
An interactive map displays high-risk areas using color-coded heatmaps.
Historical crime trends are analyzed through charts and reports.

### Patrol Route Optimization
Routes are generated from police stations to high-risk locations and back.
Dynamic routing adapts based on risk levels and station jurisdictions.
Uses a mapping API to calculate and visualize optimal patrol paths.

## Target Users
Law Enforcement: Improve patrol efficiency and crime prevention strategies.
City Planners: Analyze crime distribution for better urban planning.
Government Agencies: Allocate resources based on data-driven insights.

##Tech Stack
Frontend: React (JavaScript)
Backend: Flask (Python)
Database: Supabase (PostgreSQL)
Machine Learning: Scikit-learn, TensorFlow & PyTorch (Python)
Mapping & Routing: Google Maps API
Visualization: Google Maps JavaScript API
