import numpy as np
from sklearn.preprocessing import MinMaxScaler as _SkMinMaxScaler

dtype = np.dtype

class MinMaxScaler(_SkMinMaxScaler):
    """Alias so that pickled objects referring to MinMaxScaler.MinMaxScaler still load."""
    pass
