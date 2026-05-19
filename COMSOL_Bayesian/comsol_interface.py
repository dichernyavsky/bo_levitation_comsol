# comsol_interface.py
import numpy as np
from config import DATENSATZ, AUSDRUCK, FIXED_PARAMS

def run_one(model, combo: dict) -> tuple[np.ndarray, np.ndarray]:

    set_parameters(model, combo)
    model.solve()

    _, zeiten = model.inner(DATENSATZ)
    fx_werte = model.evaluate(AUSDRUCK, dataset=DATENSATZ)

    return np.array(zeiten, dtype=float), np.array(fx_werte, dtype=float)

def set_parameters(model, combo: dict):
    params = {
        **FIXED_PARAMS,
        "M1_H": f"{combo['M1_H']}[mm]",
        "M1_W": f"{combo['M1_W']}[mm]",
        "M2_H": f"{combo['M2_H']}[mm]",
        "M2_W": f"{combo['M2_W']}[mm]",
        "offset": f"{combo['offset']}[mm]"
    }
    for name, value in params.items():
        model.parameter(name, value)