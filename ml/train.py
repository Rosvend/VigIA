""" 
Entrenamiento de modelos
"""

def create_and_evaluate_models(X_train, X_test, y_train, y_test):
    """Create and evaluate multiple models"""

    # Define models
    models = {
        'logistic_regression': Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(max_iter=1000))
        ]),
        'random_forest': Pipeline([
            ('classifier', RandomForestClassifier(random_state=42))
        ]),
        'gradient_boosting': Pipeline([
            ('classifier', GradientBoostingClassifier(random_state=42))
        ])
    }

    # Parameter grids for each model
    param_grids = {
        'logistic_regression': {
            'classifier__C': [0.01, 0.1, 1, 10, 100],
            'classifier__penalty': ['l2'],
            'classifier__class_weight': [None, 'balanced']
        },
        'random_forest': {
            'classifier__n_estimators': [100, 200],
            'classifier__max_depth': [None, 10, 20],
            'classifier__min_samples_split': [2, 5, 10],
            'classifier__class_weight': [None, 'balanced']
        },
        'gradient_boosting': {
            'classifier__n_estimators': [100, 200],
            'classifier__learning_rate': [0.01, 0.1],
            'classifier__max_depth': [3, 5]
        }
    }

    # Train and evaluate each model
    results = {}
    best_models = {}

    for model_name, pipeline in models.items():
        print(f"\nTraining {model_name}...")

        # Grid search with cross-validation
        grid_search = GridSearchCV(
            pipeline,
            param_grids[model_name],
            cv=5,
            scoring='roc_auc',
            n_jobs=-1
        )
        grid_search.fit(X_train, y_train)

        # Get best model
        best_model = grid_search.best_estimator_
        best_models[model_name] = best_model

        # Evaluate on test set
        y_pred = best_model.predict(X_test)
        y_prob = best_model.predict_proba(X_test)[:, 1]

        # Calculate metrics
        results[model_name] = {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'test_accuracy': best_model.score(X_test, y_test),
            'test_roc_auc': roc_auc_score(y_test, y_prob),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'classification_report': classification_report(y_test, y_pred)
        }

        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best CV score: {grid_search.best_score_:.4f}")
        print(f"Test accuracy: {results[model_name]['test_accuracy']:.4f}")
        print(f"Test ROC AUC: {results[model_name]['test_roc_auc']:.4f}")
        print("Classification Report:")
        print(results[model_name]['classification_report'])

        # Save model
        joblib.dump(best_model, f"{model_name}_crime_prediction.pkl")
        print(f"Model saved as {model_name}_crime_prediction.pkl")

    # Find best overall model
    best_model_name = max(results, key=lambda k: results[k]['test_roc_auc'])
    print(f"\nBest overall model: {best_model_name}")

    return best_models, results

# Run the models
best_models, results = create_and_evaluate_models(X_train, X_test, y_train, y_test)

# Get the best model
best_model_name = max(results, key=lambda k: results[k]['test_roc_auc'])
best_model = best_models[best_model_name]

# Plot feature importance for the best model
plot_feature_importance(best_model[-1], X_train)  # -1 gets the classifier from the pipeline

# Create a prediction function
def predict_crime_probability(grid_cell, model):
    """Predict crime probability for a given grid cell"""
    # Prepare the features
    features = grid_cell.drop(['geometry', 'target'], axis=1)
    features = pd.get_dummies(features, drop_first=True)

    # Make sure features match the training data
    missing_cols = set(X_train.columns) - set(features.columns)
    for col in missing_cols:
        features[col] = 0
    features = features[X_train.columns]

    # Predict
    prob = model.predict_proba(features)[0, 1]
    return prob

# Example prediction
example_cell = crime_prediction_dataset.iloc[0:1]
prob = predict_crime_probability(example_cell, best_model)
print(f"Predicted crime probability for cell {example_cell['cell_id'].values[0]}: {prob:.4f}")

# Create a function to visualize predictions
def visualize_predictions(grid_gdf, model, X_train_cols):
    """Visualize crime predictions on map"""
    # Create a copy of the grid
    prediction_grid = grid_gdf.copy()

    # Prepare features for prediction
    features = prediction_grid.drop(['geometry', 'target'], axis=1)
    features = pd.get_dummies(features, drop_first=True)

    # Make sure features match the training data
    missing_cols = set(X_train_cols) - set(features.columns)
    for col in missing_cols:
        features[col] = 0
    features = features[X_train_cols]

    # Predict probabilities
    prediction_grid['predicted_prob'] = model.predict_proba(features)[:, 1]

    # Visualize
    fig, ax = plt.subplots(figsize=(15, 15))
    prediction_grid.plot(
        column='predicted_prob',
        cmap='YlOrRd',
        legend=True,
        ax=ax,
        legend_kwds={'label': "Crime Probability"}
    )
    ctx.add_basemap(ax, crs=prediction_grid.crs.to_string())
    plt.title('Predicted Crime Probability in Medellín')
    plt.savefig('crime_prediction_map.png')
    plt.close()

    return prediction_grid

# Visualize predictions
prediction_grid = visualize_predictions(crime_prediction_dataset, best_model, X_train.columns)
