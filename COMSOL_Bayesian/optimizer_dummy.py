# optimizer_dummy.py
import numpy as np
from datetime import datetime
from comsol_interface import run_one
from config import ALLOWED_LEFT_BOUNDRY, ALLOWED_RIGHT_BOUNDRY, W_MIN, W_MAX, H_MIN, H_MAX, OFFSET_MIN, OFFSET_MAX, AUSDRUCK, DATENSATZ
from pathlib import Path
from bayes_opt import BayesianOptimization
import optuna

RESULTS_DIR  = Path("results03")

# ----------------------
# old function

def get_valid_combinations():                                   # not used but should give you all possible valid combinations in a list of dictionarys
    valid_width_combinations = []
    for m1w in np.arange(W_MIN, W_MAX + 0.1, 0.1):
        m1w = np.round(m1w, 1)
        for m2w in np.arange(W_MIN, W_MAX + 0.1, 0.1):
            m2w = np.round(m2w, 1)
            for offset in np.arange(OFFSET_MIN, OFFSET_MAX + 0.1, 0.1):
                offset = round(offset, 1)

                ActualLB = offset - m1w
                ActualRB = offset + m2w

                if (ALLOWED_LEFT_BOUNDRY >= ActualLB) and (ActualRB >= ALLOWED_RIGHT_BOUNDRY):
                    valid_width_combinations.append((m1w, m2w, offset))

    valid_height_combinations = list(np.arange(H_MIN, H_MAX + 0.1, 0.1))
    valid_height_combinations = [round(h, 1) for h in np.arange(H_MIN, H_MAX + 0.1, 0.1)]

    valid_combinations = []
    for pair in valid_width_combinations:
        m1w = pair[0]
        m2w = pair[1]
        offset = pair[2]

        for m1h in valid_height_combinations:
            for m2h in valid_height_combinations:
                valid_combinations.append({
                    "M1_H": m1h,
                    "M1_W": m1w,
                    "M2_H": m2h,
                    "M2_W": m2w,
                    "offset": offset
                })

    return valid_combinations


# ----------------------
# might be useful functions

def combo_label(c: dict) -> str:
    label = f"M1H{c['M1_H']}_M1W{c['M1_W']}_M2H{c['M2_H']}_M2W{c['M2_W']}_OFF{c['offset']}"
    return label.replace(".", "p")                  # p for pont, 1.6 => 1 point 6 => 1p6

def save_single_result(combo: dict, zeiten: np.ndarray, fx_werte: np.ndarray):

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    fname = RESULTS_DIR / f"{combo_label(combo)}.txt"
    header = (
        f"Parameterkombination: {combo}\n"
        f"Ausdruck: {AUSDRUCK}\n"
        f"Datensatz: {DATENSATZ}\n"
        f"Erstellt: {datetime.now().isoformat()}\n"
        f"{'Zeit [s]':>16}  {'Fx [N]':>16}"
    )

    np.savetxt(
        fname,
        np.column_stack([zeiten, fx_werte]),
        fmt="%16.6e",
        header=header
    )


# ----------------------
# dummy functions

def optimize_dummy01(model):        # dummy function, just for check if the rest is working
    
    combination = {             # exampel hard coded, the run_one() function needs a dictionary like that
            "M1_H": 5,    
            "M1_W": 3.5,  
            "M2_H": 7,    
            "M2_W": 4.0,  
            "offset": 0.5 
        }

    results = []

    try:
        time, fx_val = run_one(model, combination)                    # zeiten = time; fx_werte = fx_val

        save_single_result(combination, time, fx_val)                 # this creats you a folder result with the single results
        print(combination, f" Zeit: {time[-1]} ", f"Fx_Ende: {fx_val[-1]} [N]")
        
        results.append((combination, float(fx_val[-1]), time[-1]))    # currently just the last entry is important, it is the maximum

    except Exception as e:
        print(f"Fehler bei Kombination {combination}: {e}")


    return #smth


def optimize_dummy02(model, n_trials=50):   # dummy function, just to check if the rest is working
 
    def objective(trial):
        m1w    = trial.suggest_float("M1_W",   W_MIN,      W_MAX,      step=0.1)
        m2w    = trial.suggest_float("M2_W",   W_MIN,      W_MAX,      step=0.1)    # maybe use W2_MAX as a function of W_MAX --> W2_Max = 15 - W_Max?
        offset = trial.suggest_float("offset", OFFSET_MIN, OFFSET_MAX, step=0.1)
        m1h    = trial.suggest_float("M1_H",   H_MIN,      H_MAX,      step=0.1)
        m2h    = trial.suggest_float("M2_H",   H_MIN,      H_MAX,      step=0.1)
 
        ActualLB = offset - m1w
        ActualRB = offset + m2w
        if not (ALLOWED_LEFT_BOUNDRY <= ActualLB and ActualRB <= ALLOWED_RIGHT_BOUNDRY):
            return -1e9
 
        combo = {
            "M1_H":   round(m1h,    1),
            "M1_W":   round(m1w,    1),
            "M2_H":   round(m2h,    1),
            "M2_W":   round(m2w,    1),
            "offset": round(offset, 1),
        }
 
        try:
            time, fx_val = run_one(model, combo)
            save_single_result(combo, time, fx_val)
            print(combo, f" Zeit: {time[-1]} ", f"Fx_Ende: {fx_val[-1]} [N]")
            return float(fx_val[-1])
        except Exception as e:
            print(f"Fehler bei Kombination {combo}: {e}")
            return -1e9
 
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
 
    best = study.best_trial
    best_combo = {
        "M1_H":   round(best.params["M1_H"],   1),
        "M1_W":   round(best.params["M1_W"],   1),
        "M2_H":   round(best.params["M2_H"],   1),
        "M2_W":   round(best.params["M2_W"],   1),
        "offset": round(best.params["offset"], 1),
    }
 
    print(f"\nBeste Kombination: {best_combo}")
    print(f"Bestes Fx: {best.value:.4f} N")
 
    return best_combo

