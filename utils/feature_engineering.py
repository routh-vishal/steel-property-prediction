from sklearn.preprocessing import FunctionTransformer
features = ['C','Mn','Cr','Ni','Al','Temperature_C']
def enhanced_features(X):
    X_new = X.copy()
    X_new['Temperature_C_squared'] = X_new[features[-1]] ** 2
    for elem in features[0:-1]:
        X_new[f'Temperature_C_x_{elem}'] = X_new[features[-1]] * X_new[elem]
    return X_new