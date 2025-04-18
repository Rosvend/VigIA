from pandas import DataFrame, Series
from numpy import ndarray

class ModelWrapperInterface:
    """
    Informal interface for classes that abstract feature preprocessing
    and model predicton.

    Attributes
    ----------
    _model_data: dict
        The dictionary containing the model and other relevant objects.
    _grid_features: DataFrame
        DataFrame of cell features, including their indexes.
    """

    _model_data: dict
    _grid_features: DataFrame

    def __init__(self, model_data_path: str, grid_features_path: str):
        """
        Initialize the wrapper.
        
        Parameters
        ----------
        model_data_path: str
            The path of the pickle file that contains the
            model and related data.
        grid_features_path: str
            The path of the csv file that contains the
            cell feature data.
        """
        pass

    def predict(self, grid_cells: DataFrame) -> DataFrame:
        """
        Predict the crime score for the given grid cells.

        Parameters
        ----------
        grid_cells: DataFrame
            A pandas DataFrame containing the h3_index of the cells to
            predict probabilities for.
        
        
        Returns
        -------
        DataFrame
            The DataFrame original dataframe, with the column probability
            added.
        """
        pass
