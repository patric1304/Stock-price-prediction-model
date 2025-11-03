import numpy as np

def add_delta_features(X):
    """
    Add simple delta (day-to-day change) features to the stock time series portion.
    """
    X_new = X.copy()
    # Assuming first 5*HISTORY_DAYS elements are OHLCV
    window_len = 5  # OHLCV
    ts_len = (X.shape[1] - 6)  # remaining features
    for i in range(X.shape[0]):
        ts = X[i, :window_len*20].reshape(20, window_len)  # last 20 days
        delta = ts[1:] - ts[:-1]
        delta_flat = delta.flatten()
        X_new[i] = np.concatenate([X[i,:], delta_flat])
    return X_new
