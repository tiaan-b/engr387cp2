import matplotlib.pyplot as plt

from src import lodecci

def main():
    # --- SET PARAMETERS ---
    m = 1
    c = 0.5
    k = 4
    
    dt = 0.001
    f = None
    ic = [2,0]
    t0 = 0.0
    
    tStop = 10
    # ----------------------
    
    model = lodecci.LODECCI_IVP_Model([k,c,m], dt, f = f, ic = ic, t0 = t0)
    
    x, t = model.eval(tStop)

    fig, ax = plt.subplots()
    ax.plot(t, x, label="x(t)")
    ax.legend()
    
    plt.show()

if __name__ == "__main__":
    main()