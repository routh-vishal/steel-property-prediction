from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_error

def evaluate_model(y_true, y_pred, name):
    r2 = r2_score(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    print(f"\n{name}:")
    print(f"R2: {r2:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}")
    return r2, rmse, mae