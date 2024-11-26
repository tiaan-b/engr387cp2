import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from typing import Iterable, Callable

class LODECCI_IVP_Model:
    def __init__(
            self,
            coeffs: Iterable[float],
            dt: float,
            f: Callable[[np.float64],np.float64] | None = None,
            ic: Iterable[float] | float = 1.0,
            t0: float = 0.0,
            fLatex: str | None = None,
            analyticalSolution: Callable[[np.ndarray],np.ndarray] | None = None
            ):
        
        if len(coeffs) == 0:
            raise ValueError("Must specify at least one coefficient")
        if coeffs[-1] == 0:
            raise ValueError("The last element in coeffs must be nonzero")
        
        def anaSolPlaceholder():
            raise NotImplementedError("Analytical solution is not specified.")
        
        self._coeffs = np.array(coeffs, dtype = np.float64)
        self._n = len(self._coeffs) - 1
        self._f = f if f is not None else lambda t: np.float64(0.0)
        self._dt = np.float64(dt)
        self._ic = np.array(
            ic if isinstance(ic, Iterable) else [ic] * self._n,
            dtype = np.float64
        )
        self._t0 = np.float64(t0)
        self._fLatex = (
            fLatex if fLatex is not None 
            else "0" if f is None 
            else "f(t)"
        )
        self.analyticalSolution = (
            anaSolPlaceholder if analyticalSolution is None
            else analyticalSolution
        )
        
        if not len(self._ic) == self._n:
            raise ValueError(f"Must specify {self._n} initial conditions.")
        
        # Construct the model
        A = np.vstack(
            (
                np.eye(self._n - 1, self._n, 1, dtype = np.float64), 
                self._coeffs[:-1] * -1 / self._coeffs[-1]
            ),
            dtype = np.float64
        )
        B = self._dt * A + np.eye(self._n)
        def step(xi: np.ndarray, i: int) -> np.ndarray:
            xip1 = B @ xi
            fi = self._f(self._t0 + i * self._dt)
            xip1[-1] = xip1[-1] + self._dt * fi / self._coeffs[-1]
            return xip1
        self._step = step
        
    @property
    def coeffs(self) -> np.ndarray:
        return self._coeffs.copy()
    
    @property
    def ic(self) -> np.ndarray:
        return self._ic.copy()
    
    @property
    def t0(self) -> np.float64:
        return self._t0.copy()
        
    @property
    def eom(self):
        eq = ""
        for i in range(len(self._coeffs)):
            coeff = self._coeffs[i]
            if coeff > 0:
                sign = "+"
            elif coeff < 0:
                sign = "-"
            else:
                continue
            
            eq = sign + f"{abs(float(coeff))}x^{{({i})}}(t)" + eq
            
        if eq[0] == "+":
            eq = eq[1:]
        eq = eq.replace("^{(0)}", "")
        eq = "$" + eq + "=" + self._fLatex + "$"
        return eq
        
    def eval(
            self,
            tStop: float,
            tStart: float | None = None,
            includeEndpoints: tuple[bool, bool] = (True, True),
            zerothOrderOnly: bool = True
        ) -> tuple[np.ndarray, np.ndarray]:
        
        tStart = self._t0 if tStart is None else np.float64(tStart)
        tStop = np.float64(tStop)
        if tStop < tStart:
            raise ValueError(
                f"tStop ({tStop})cannot be less than tStart ({tStart})."
            )
            
        roundStart = np.floor if includeEndpoints[0] else np.ceil
        iStart = int(roundStart((tStart - self._t0) / self._dt))
        roundStop = np.ceil if includeEndpoints[1] else np.floor
        iStop = int(roundStop((tStop - self._t0) / self._dt))
        
        inds = np.arange(iStop + 1)
        out = np.empty((iStop + 1, self._n), dtype=np.float64)
        out[0, :] = self._ic
        for i in inds[:-1]:
            out[i + 1, :] = self._step(out[i, :], i)
            
        if zerothOrderOnly:
            out = out[:, 0]
        out = out[iStart:, ...]
        inds = inds[iStart:]
        
        return out, inds * self._dt + self._t0  
        
    def evalAt(
            self,
            t: float
        ) -> tuple[np.ndarray, np.ndarray]:
        
        return self.eval(t, tStart = t)
    
    def quickPlot(
            self, 
            *args, 
            ax: matplotlib.axes._axes.Axes | None = None,
            showPlot: bool = True,
            plotAnalyticalSolution: bool = False,
            **kwargs,
            ) -> tuple[np.ndarray, np.ndarray, matplotlib.axes._axes.Axes]:
            
        x, t = self.eval(*args, **kwargs)
            
        if ax is None:
            _, ax = plt.subplots()
                
        n = 1 if not len(x.shape) > 1 else x.shape[1]
        labels = [f"$x^{{({i})}}(t)$" for i in range(n)]
        labels[0] = "$x(t)$"
                
        ax.plot(t, x, label = labels)
        
        x_analytical = None
        if plotAnalyticalSolution:
            x_analytical = self.analyticalSolution(t)
            ax.plot(t, x_analytical, label = "Analytical Solution $x_{a}(t)$")
        
        ax.set_title(self.eom + f", $\Delta t = {self._dt}$")
        ax.set_xlabel("$t$")
        ax.legend()
            
        if showPlot:
            plt.show()
                
        return x, t, ax, x_analytical
    
    
class MSDp_cosineForcing_IVP_Model(LODECCI_IVP_Model):
    # For systems of the form: m*x''(t) + c*x'(t) + k*x(t) = F_0*cos(omega*t)
    # Values specified for f and fLatex are ignored
    def __init__(
            self,
            m: float,
            c: float,
            k: float,
            F_0: float,
            omega: float,
            dt: float,
            **kwargs
            ):
        
        kwargs["fLatex"] = f"{F_0}\cos{{({omega}t)}}"
        omega = np.float64(omega)
        F_0 = np.float64(F_0)
        kwargs["f"] = lambda t: F_0 * np.cos(omega * t)
        super().__init__(
            [k,c,m],
            dt,
            **kwargs
        )
            
        self._m = np.float64(m)
        self._c = np.float64(c)
        self._k = np.float64(k)
        self._F_0 = F_0
        self._omega = omega
        
    @property
    def F_0(self) -> np.float64:
        return self._F_0.copy()
    
    @property
    def omega(self) -> np.float64:
        return self._omega.copy()
        
    def getHomogenousModel(self) -> LODECCI_IVP_Model:
        return LODECCI_IVP_Model(
            [self._k,self._c,self._m],
            self._dt,
            f = None,
            ic = self._ic,
            t0 = self._t0
        )