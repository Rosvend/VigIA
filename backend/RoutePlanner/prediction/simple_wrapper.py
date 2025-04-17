from .wrapper_interface import ModelWrapperInterface
import logging
import sys
from pandas import DataFrame
from pandas import Series
import joblib
from datetime import datetime
from numpy import ndarray
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

sys.path.append("../ml")

from quick_predict import generate_quick_predictions

#logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SimpleModelWrapper(ModelWrapperInterface):
    """
    A class to wrap a Simple/Quick model and obtain predictions from it.
    """

    _model: RandomForestClassifier
    _scaler: StandardScaler
    _numerical_columns: list
    _required_features: list

    def __init__(self, model_data_path: str, grid_features_path: str):
        
        # Loading the model and related data
        logger.info(f"Loading model from {model_data_path}")
        self._model_data = joblib.load(model_data_path)
        logger.info(f"Model loaded successfully - trained on {self._model_data.get('training_date', 'unknown date')}")

        self._model = self._model_data['model']
        self._scaler = self._model_data['scaler']
        self._numerical_columns = self._model_data['numerical_columns']
        self._required_features = self._model_data.get('features', [])

        # Load the features dataframe into the object
        self._grid_features = pd.read_csv(grid_features_path)

    def predict(self, grid_cells: DataFrame) -> DataFrame:
        df = generate_quick_predictions(self._model_data, grid_cells)
        df['probability'] = df['risk']
        df = df.drop('risk', axis=1)
        return df
