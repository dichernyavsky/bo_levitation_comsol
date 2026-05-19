# main.py
import mph
from optimizer_dummy import optimize_dummy02    #RENAME Function Import

def main():
    client = mph.start()    # its important to have this on a "highl-level" because the client needs time to start
    model = client.load("3D_H-phi_PM-Pair_N38_ld1_z4_V3.mph")

    print(optimize_dummy02(model))  #RENAME Function

    #some stuff

    client.clear()

if __name__ == "__main__":
    main()