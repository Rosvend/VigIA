from .wrapper_interface import ModelWrapperInterface
from datetime import datetime, timedelta
from pandas import DataFrame
import joblib
import logging
import pandas as pd
import sys

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
    UPDATE_CACHE_TIMEDIFF = timedelta(hours=24)
    _cell_predictions_cache: DataFrame
    _last_updated: datetime

    def __init__(self, model_data_path: str):
        
        # Loading the model and related data
        logger.info(f"Loading model from {model_data_path}")
        self._model_data = joblib.load(model_data_path)
        logger.info(f"Model loaded successfully - trained on {self._model_data.get('training_date', 'unknown date')}")

        # Generate the predictions and store them on cache
        self.update_prediction_cache()

    def update_prediction_cache(self):
        self._cell_predictions_cache = generate_quick_predictions(self._model_data)
        self._last_updated = datetime.now()

    def predict(self, grid_cells: DataFrame) -> DataFrame:
        if (datetime.now() - self._last_updated) >= self.UPDATE_CACHE_TIMEDIFF:
            self.update_prediction_cache()
        df = pd.merge(grid_cells, self._cell_predictions_cache[['h3_index', 'risk']], on='h3_index')
        df['probability'] = df['risk']
        df = df.drop('risk', axis=1)
        return df
