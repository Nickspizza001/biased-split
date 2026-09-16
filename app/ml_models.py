import numpy as np
import inspect
from typing import Any
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


def _split_for_single_bias(splitter_obj, smiles_list, activities, bias, random_seed=42):
    """Normalize the splitter API across all splitters used in this project."""
    split_for_intended: Any = getattr(splitter_obj, "split_for_intended_bias", None)
    if callable(split_for_intended):
        params = inspect.signature(split_for_intended).parameters
        if len(params) >= 4 and "intended_bias" in params:
            try:
                result = split_for_intended(
                    smiles_list,
                    activities,
                    bias,
                    random_seed=random_seed,
                )
                if isinstance(result, tuple) and len(result) == 3:
                    return result
                if isinstance(result, tuple) and len(result) == 5:
                    return result[0], result[1], result[2]
            except TypeError:
                pass

    split_method: Any = getattr(splitter_obj, "split", None)
    if callable(split_method):
        try:
            try:
                result = next(
                    split_method(smiles_list, activities, intended_biases=[bias], n_repeats=1)
                )
            except TypeError:
                result = split_method(smiles_list, activities, intended_bias=bias)
            if isinstance(result, tuple) and len(result) == 5:
                return result[0], result[1], result[2]
            if isinstance(result, tuple) and len(result) == 3:
                return result
        except TypeError:
            pass

        try:
            result = split_method(smiles_list, activities)
            if isinstance(result, tuple) and len(result) == 3:
                return result
            if isinstance(result, tuple) and len(result) == 5:
                return result[0], result[1], result[2]
        except TypeError:
            pass

        try:
            result = split_method(smiles_list)
            if isinstance(result, tuple) and len(result) == 3:
                return result
            if isinstance(result, tuple) and len(result) == 5:
                return result[0], result[1], result[2]
        except TypeError:
            pass

    raise TypeError(
        f"Splitter {type(splitter_obj).__name__} does not support a single intended_bias split API."
    )

def train_and_eval_regressor(X_train, y_train, X_test, y_test, model_type="Random Forest"):
    if model_type == "Random Forest":
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    else:
        model = Ridge(alpha=1.0)

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    metrics = {
        "R2": float(r2_score(y_test, preds)),
        "RMSE": float(np.sqrt(mean_squared_error(y_test, preds))),
        "MAE": float(mean_absolute_error(y_test, preds))
    }
    return model, preds, metrics

def run_bias_sweep(smiles_list, activities, fps, splitter_obj, biases, model_type="Random Forest"):
    results = []
    for b in biases:
        tr_idx, te_idx, eff_bias = _split_for_single_bias(splitter_obj, smiles_list, activities, b)
        _, _, metrics = train_and_eval_regressor(
            fps[tr_idx], activities[tr_idx], fps[te_idx], activities[te_idx], model_type=model_type
        )
        results.append({
            "Intended Bias": b,
            "Effective Bias": eff_bias,
            "R2": metrics["R2"],
            "RMSE": metrics["RMSE"],
            "MAE": metrics["MAE"],
            "Train Size": len(tr_idx),
            "Test Size": len(te_idx)
        })
    return results