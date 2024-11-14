import numpy as np
from typing import Iterable, Callable

class LODECCI_IVP_Model:
    def __init__(
            self,
            coeffs: Iterable[float],
            dt: float,
            f: Callable[[np.float64], np.float64] | None = None,
            ic: Iterable[float] | float = 0.0,
            t0: float = 0.0,
            ):
        
        # TODO: check lengths of n
        self._coeffs = np.array(coeffs, dtype = np.float64)
        self._n = len(self._coeffs) - 1
        self._f = f if f is not None else lambda t: np.float64(0.0)
        self._dt = np.float64(dt)
        self._ic = np.array(
            ic if isinstance(ic, Iterable) else [ic] * self._n,
            dtype = np.float64
        )
        self._t0 = np.float64(t0)
        
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
        
    def eval(
            self,
            tStop: float,
            tStart: float | None = None,
            includeEndpoints: tuple[bool, bool] = (True, True),
            zerothOrderOnly: bool = True
        ):
        
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
        ):
        
        return self.eval(t, tStart = t)