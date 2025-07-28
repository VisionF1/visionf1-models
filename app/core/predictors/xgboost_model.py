class XGBoostPredictor:
    def __init__(self, params=None):
        import xgboost as xgb
        self.model = xgb.XGBRegressor(**(params if params else {}))

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)