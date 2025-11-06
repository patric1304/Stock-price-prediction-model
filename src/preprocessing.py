from sklearn.preprocessing import StandardScaler

def scale_features(X, y=None):
    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    if y is not None:
        scaler_y = StandardScaler()
        y_scaled = scaler_y.fit_transform(y.reshape(-1,1))
        return X_scaled, y_scaled, scaler_X, scaler_y
    return X_scaled, scaler_X