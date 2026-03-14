"""
Régression OLS avec tests statistiques (Student, p-value, IC) via statsmodels.

Utilisé par tous les modèles Markowitz (1 facteur, 3 facteurs, 5 facteurs, LLM)
pour exposer la significativité des facteurs (p-value, t-stat, etc.).
"""
from __future__ import annotations

from typing import Any

import numpy as np
import statsmodels.api as sm


def ols_factor_regression(
    y: np.ndarray,
    X_factors: np.ndarray,
    factor_names: list[str],
    *,
    add_constant: bool = True,
) -> dict[str, Any] | None:
    """
    Régression OLS (y sur constante + facteurs) avec statsmodels.

    Args:
        y: vecteur des rendements en excès (n,)
        X_factors: matrice des rendements des facteurs (n, k), en décimal (ex: 0.01 pour 1%)
        factor_names: noms des colonnes de X_factors (ordre identique)
        add_constant: si True, une constante (intercept) est ajoutée

    Returns:
        Dict avec:
          - "coeffs": {"alpha": float, "factor1": float, ...} pour compatibilité
          - "factor_tests": {"factor1": {"beta", "std_err", "t_stat", "p_value", "ci_lower", "ci_upper"}, ...}
            plus "alpha" pour l'intercept si add_constant
          - "model_stats": {"r_squared", "adj_r_squared", "n_obs", "df_residual", "f_stat", "f_pvalue"}
        Ou None si la régression échoue (matrice singulière, etc.).
    """
    n, k = X_factors.shape
    if n < k + (2 if add_constant else 1) or len(factor_names) != k:
        return None

    X = sm.add_constant(X_factors, has_constant="add") if add_constant else X_factors
    try:
        model = sm.OLS(y, X).fit()
    except Exception:
        return None

    # Coefficients pour compatibilité avec le code existant
    # model.params / bse / etc. peuvent être Series ou ndarray selon la version statsmodels
    def _at(obj, i):
        return obj.iloc[i] if hasattr(obj, "iloc") else obj[i]

    coeffs: dict[str, float] = {}
    if add_constant:
        coeffs["alpha"] = float(_at(model.params, 0))
    for i, name in enumerate(factor_names):
        idx = i + (1 if add_constant else 0)
        coeffs[name] = float(_at(model.params, idx))

    # Tests par coefficient (alpha + facteurs)
    factor_tests: dict[str, dict[str, float]] = {}
    param_names = list(model.params.index) if hasattr(model.params, "index") else list(range(len(model.params)))
    ci_df = model.conf_int(alpha=0.05)
    for i, pname in enumerate(param_names):
        key = "alpha" if (add_constant and i == 0) else factor_names[i - 1] if add_constant else factor_names[i]
        ci_row = _at(ci_df, i)
        ci_low = float(ci_row.iloc[0]) if hasattr(ci_row, "iloc") else float(ci_row[0])
        ci_high = float(ci_row.iloc[1]) if hasattr(ci_row, "iloc") else float(ci_row[1])
        factor_tests[key] = {
            "beta": round(float(_at(model.params, i)), 6),
            "std_err": round(float(_at(model.bse, i)), 6),
            "t_stat": round(float(_at(model.tvalues, i)), 4),
            "p_value": round(float(_at(model.pvalues, i)), 6),
            "ci_lower": round(ci_low, 6),
            "ci_upper": round(ci_high, 6),
        }

    model_stats = {
        "r_squared": round(float(model.rsquared), 6),
        "adj_r_squared": round(float(model.rsquared_adj), 6),
        "n_obs": int(model.nobs),
        "df_residual": int(model.df_resid),
        "f_stat": round(float(model.fvalue), 4) if model.fvalue is not None else None,
        "f_pvalue": round(float(model.f_pvalue), 6) if model.f_pvalue is not None else None,
    }

    return {
        "coeffs": coeffs,
        "factor_tests": factor_tests,
        "model_stats": model_stats,
    }
