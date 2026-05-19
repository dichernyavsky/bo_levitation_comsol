"""
----------------------------------------------------
HTS Parameterstudie für Optimierung  |  mph 1.3.1  |  
----------------------------------------------------
"""

import mph
import numpy as np
import random
import threading
import logging
from pathlib import Path
from datetime import datetime

import tkinter as tk
from tkinter import ttk
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# ==============================================================================
# Konfigurationen
# ==============================================================================

MODEL_PATH   = Path("3D_H-phi_PM-Pair_N38_ld1_z4_V3.mph")
RESULTS_DIR  = Path("ergebnisse03")
DATENSATZ    = "Studie 1//Lösung 1"
AUSDRUCK     = "intop1(Jy*mfh.Bz-mfh.By*Jz)"    #this is the optimization parameter (Fx in N)
N_KOMBINATIONEN      = 3        # Anzahl zufälliger Parameterkombinationen


# Feste Parameter
FIXED_PARAMS = {
    "cooling_gap":    "4[mm]",                  # from 3 to 5. but it would be better to have an optimization for 3 mm, one for 4 mm and one for 5 mm. depending on what cooling gap ist achievable you can chose your magnet arrangement.
    "M_length":       "30[mm]",                 # 30 mm are realistic for measurement but can deviat
    "verschiebung":   "1",                      # currently not used                      
    "Ec":             "1e-4[V/m]",              # material parameter
}


# Grenzen für variable Parameter
GW_MAX = 15        # M1_W + M2_W [mm]
#GW_MIN = 6        # dont need this one
W_MIN  = 3         # Mindestbreite [mm] / min width [mm]
W_MAX  = 12        # Maximalbreite [mm] / max width [mm]
H_MIN  = 3         # Mindesthöhe  [mm] / min height [mm]
H_MAX  = 15        # Maximalhöhe  [mm] / max height [mm]
OFFSET_MIN  = -GW_MAX/2 + W_MIN  # -4.5 # [mm], offset / joint_position
OFFSET_MAX  = -OFFSET_MIN        # 4.5  # [mm], offset / joint_position


# ==============================================================================
# Zufallskombinationen
# ==============================================================================

def generate_random_combinations(n: int) -> list[dict]:

    #gültigen Kombinationen
    valid_width_combinations = []
    for m1w in np.arange(W_MIN, W_MAX + 0.1, 0.1):
        m1w = np.round(m1w, 1)
        for m2w in np.arange(W_MIN, W_MAX + 0.1, 0.1):
            m2w = np.round(m2w, 1)
            for offset in np.arange(OFFSET_MIN, OFFSET_MAX + 0.1, 0.1):
                offset = round(offset, 1)

                AllowedLeftBoundry = -GW_MAX/2
                AllowedRightBoundry = -AllowedLeftBoundry
                AcutalLB = offset - m1w 
                ActualRB = offset + m2w
                
                if (AllowedLeftBoundry >= AcutalLB) and (ActualRB >= AllowedRightBoundry):
                    valid_width_combinations.append((m1w, m2w, offset))

    valid_height_combinations = list(range(H_MIN, H_MAX + 1))

    #Suchraum
    suchraum = []
    for pair in valid_width_combinations:
        m1w = pair[0]  # Breite von Magnet 1
        m2w = pair[1]  # Breite von Magnet 2
        offset = pair[2]

        for m1h in valid_height_combinations:
            for m2h in valid_height_combinations:
                suchraum.append({
                    "M1_H": m1h,
                    "M1_W": m1w,
                    "M2_H": m2h,
                    "M2_W": m2w,
                    "offset" : offset
                })

    #zufällige Kombinationen
    if n > len(suchraum):
        n = len(suchraum)
    chosen = random.sample(suchraum, n)

    return chosen

def combo_label(c: dict) -> str:
    label = f"M1H{c['M1_H']}_M1W{c['M1_W']}_M2H{c['M2_H']}_M2W{c['M2_W']}_OFF{c['offset']}"
    return label.replace(".", "p")                  # p for pont, 1.6 => 1 point 6 => 1p6

# ==============================================================================
# Simulation
# ==============================================================================

def set_parameters(model, combo: dict):
    """Übergabe an COMSOL :)"""
    params = {
        **FIXED_PARAMS,
        "M1_H"   : f"{combo['M1_H']}[mm]",
        "M1_W"   : f"{combo['M1_W']}[mm]",
        "M2_H"   : f"{combo['M2_H']}[mm]",
        "M2_W"   : f"{combo['M2_W']}[mm]",
        "offset" : f"{combo['offset']}[mm]"
    }
    for name, value in params.items():
        model.parameter(name, value)

def run_one(model, combo: dict) -> tuple[np.ndarray, np.ndarray]:

    set_parameters(model, combo)
    model.solve()

    _, zeiten = model.inner(DATENSATZ)
    fx_werte = model.evaluate(AUSDRUCK, dataset=DATENSATZ)

    zeiten   = np.array(zeiten,   dtype=float)
    fx_werte = np.array(fx_werte, dtype=float)

    #---- Datei speichern -------------------------------------------------------
    fname = RESULTS_DIR / f"{combo_label(combo)}.txt"
    header = (
        f"Parameterkombination: {combo}\n"
        f"Ausdruck: {AUSDRUCK}\n"
        f"Datensatz: {DATENSATZ}\n"
        f"Erstellt: {datetime.now().isoformat()}\n"
        f"{'Zeit [s]':>16}  {'Fx [N]':>16}"
    )
    np.savetxt(fname, np.column_stack([zeiten, fx_werte]),
               fmt="%16.6e", header=header)
    #---------------------------------------------------------------------------

    return zeiten, fx_werte

def simulation_thread(Kombinationen: list[dict]):
    client = mph.start()
    model = client.load(str(MODEL_PATH))

    for index, combo in enumerate(Kombinationen, start=1):
        print(index, combo, "Simulation läuft …")

        try:
            zeiten, fx_werte = run_one(model, combo)
            fx_end  = float(fx_werte[-1])
            zeit = zeiten[-1]

            print(combo_label(combo))
            print(fx_end)
            print(f"Zeit: {zeit} ", f"Fx_Ende: {fx_end} [N]")

        except Exception as e:
            print(f"Fehler bei {combo_label(combo)}: {e} :())")

    client.clear()

# ==============================================================================
# MAIN
# ==============================================================================

def main():

    Kombinationen = generate_random_combinations(N_KOMBINATIONEN)

    print("Geplante Parameterkombinationen:")
    for i, c in enumerate(Kombinationen, 1):
        print(f"  #{i:2d}  {combo_label(c)}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    simulation_thread(Kombinationen)
    

if __name__ == "__main__":
    main()