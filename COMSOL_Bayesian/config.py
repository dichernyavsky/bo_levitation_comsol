# config.py
DATENSATZ = "Studie 1//Lösung 1"
AUSDRUCK = "intop1(Jy*mfh.Bz-mfh.By*Jz)"    #this is the optimization parameter (Fx in N)
FIXED_PARAMS = {
    "cooling_gap": "4[mm]",                 # from 3 to 5. but it would be better to have an optimization for 3 mm, one for 4 mm and one for 5 mm. depending on what cooling gap ist achievable you can chose your magnet arrangement.
    "M_length": "30[mm]",                   # 30 mm are realistic for measurement but can deviat
    "verschiebung": "1",                    # currently not used 
    "Ec": "1e-4[V/m]"                       # material parameter
}

# Grenzen für variable Parameter
#GW_MAX = 15        # M1_W + M2_W [mm]
#GW_MIN = 6        # dont need this one
ALLOWED_LEFT_BOUNDRY = -7.5                #NEW i think its better than GW (total width)
ALLOWED_RIGHT_BOUNDRY = -ALLOWED_LEFT_BOUNDRY       #NEW

W_MIN  = 3         # Mindestbreite [mm] / min width [mm]
W_MAX  = 12        # Maximalbreite [mm] / max width [mm]
H_MIN  = 3         # Mindesthöhe  [mm] / min height [mm]
H_MAX  = 15        # Maximalhöhe  [mm] / max height [mm]
OFFSET_MIN  = ALLOWED_LEFT_BOUNDRY + W_MIN  # -4.5 # [mm], offset / joint_position
OFFSET_MAX  = -OFFSET_MIN        # 4.5  # [mm], offset / joint_position

