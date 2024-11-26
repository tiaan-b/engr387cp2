from typing import Callable

import matplotlib.pyplot as plt
import numpy as np

from src import lodecci

def main():
    # Answer all the questions in course project 2
        
    problems = getProblemSet(constructAnalyticalSolution = True)
    for i, problem in enumerate(problems):
        models, xs, ts, axs, x_analyticals = problem(showPlot = False)
        fig = axs[0].get_figure()
        fig.suptitle(f"Problem {i + 1}")
        
        if i + 1 == 6:
            for ax in axs:
                ax.title.set_size(0.8 * ax.title.get_size())
        
    plt.show()

def getProblemSet(*args, **kwargs):
    problemParams = getProblemParams(*args, **kwargs)
    problems = []

    for subProblemParams in problemParams:
        subProblems = [
            getGenericProblem(*p[0], **p[1]) for p in subProblemParams
        ]
        problems.append(packageProblems(subProblems))
        
    return problems  
    
# Constructs problems for mass-spring-dashpot systems, optionally with a cosine
# forcing term.   
def getGenericProblem(
        tStart,
        tStop,
        *args,
        homogenousSolutionOnly: bool = True,
        constructAnalyticalSolution: bool = False,
        **kwargs
        ):
    
    def p(ax = None, showPlot = False):
        
        analyticalSolutionSpecified = "analyticalSolution" in kwargs
        
        model = lodecci.MSDp_cosineForcing_IVP_Model(*args, **kwargs)
        if homogenousSolutionOnly:
            model = model.getHomogenousModel()
        
        if not analyticalSolutionSpecified and constructAnalyticalSolution:
            # Construct the analytical solution (if possible)
            
            niErr = lambda reason: NotImplemented(
                "The analytical solution cannot be automatically constructed "
                + "as the following is not supported: " + reason
            )
            
            k, c, m = model.coeffs
            x_0, v_0 = model.ic
            F_0, omega = (
                [model.F_0, model.omega] if not homogenousSolutionOnly 
                else [None] * 2
            )
            
            if model.t0 != np.float64(0):
                raise niErr("Initial conditions not at t=0 (ie. t0 != 0)")
            
            omega_n = np.sqrt(k / m)
            c_c = 2 * np.sqrt(k * m)
            zeta = c / c_c
            omega_d = omega_n * np.sqrt(1 - np.power(zeta, 2))          
            
            atResonance = omega == omega_n
            
            # For clarity, xp_0 is used instead of X_0 
            if zeta == 0:
                # Undamped case (c = 0)
                if homogenousSolutionOnly:
                    A = x_0
                    B = v_0 / omega_n
                    x_g = lambda t: (
                        A * np.cos(omega_n * t)
                        + B * np.sin(omega_n * t)
                    )
                elif atResonance:
                    xp_0 = (F_0 * omega_n) / (2 * k)
                    phi = np.pi / 2
                    A = x_0
                    B = v_0 / omega_n
                    x_g = lambda t: (
                        A * np.cos(omega_n * t)
                        + B * np.sin(omega_n * t)
                        + xp_0 * t * np.sin(omega_n * t)
                    )
                else:
                    xp_0 = F_0 / (k - m * np.power(omega, 2))
                    phi = np.float64(0)
                    A = x_0 - xp_0
                    B = v_0 / omega_n
                    x_g = lambda t: (
                        A * np.cos(omega_n * t)
                        + B * np.sin(omega_n * t)
                        + xp_0 * np.cos(omega * t)
                    )
            elif zeta < 1 and zeta > 0:
                # Underdamped case
                if homogenousSolutionOnly:
                    A = x_0
                    B = (v_0 + zeta * omega_n * x_0) / omega_d
                    x_g = lambda t : (
                        np.exp(-zeta * omega_n * t) * (
                            A * np.cos(omega_d * t) + B * np.sin(omega_d * t)
                        )
                    )
                elif atResonance:
                    xp_0 = F_0 / (c * omega_n)
                    phi = np.pi / 2
                    A = x_0
                    B = (v_0 + zeta * omega_n * x_0 - F_0 / c) / omega_d
                    x_g = lambda t : (
                        np.exp(-zeta * omega_n * t) * (
                            A * np.cos(omega_d * t) + B * np.sin(omega_d * t)
                        )
                        + xp_0 * np.sin(omega_n * t)
                    )
                else:
                    xp_0 = F_0 / np.sqrt(
                        np.power((k - m * np.power(omega, 2)), 2)
                        + np.power((c * omega), 2)
                    )
                    phi = np.arctan((c * omega) / (k - m * np.power(omega, 2)))
                    A = x_0 - xp_0 * np.cos(phi)
                    B = (
                        (
                            v_0
                            + zeta * omega_n * (x_0 - xp_0 * np.cos(phi))
                            - xp_0 * omega * np.sin(phi)
                        )
                        / omega_d
                    )
                    x_g = lambda t : (
                        np.exp(-zeta * omega_n * t) * (
                            A * np.cos(omega_d * t) + B * np.sin(omega_d * t)
                        )
                        + xp_0 * np.cos(omega * t - phi)
                    )
            elif zeta == 1:
                # Critically damped case
                raise niErr("Critically damped systems")
            elif zeta > 1:
                # Overdamped case
                raise niErr("Overdamped systems")
            else:
                raise niErr(
                    "Zeta < 0 (This should not be possible for physical "
                    + "systems, check model parameters)"
                )
            
            model.analyticalSolution = x_g
            analyticalSolutionSpecified = True
            
        x, t, axOut, x_analytical = model.quickPlot(
            tStop,
            tStart = tStart,
            zerothOrderOnly = True,
            ax = ax,
            showPlot = showPlot,
            plotAnalyticalSolution = analyticalSolutionSpecified
        )
        
        return model, x, t, axOut, x_analytical
    
    return p

def packageProblems(problems):
    def p(axs = None, showPlot = False):
        if axs is None:
            _, axs = plt.subplots(
                ncols = len(problems), 
                squeeze = False, 
                sharey = True,
                layout = "constrained"
            )
    
        out = []
        for problem, ax in zip(problems, axs.flatten()):
            out.append(problem(ax = ax, showPlot = showPlot))
        out = [list(x) for x in zip(*out)]

        return tuple(out)
    
    return p

# Problem parameters for mass-spring-dashpot systems, optionally with a cosine
# forcing term.   
def getProblemParams(
        homogenousSolutionOnly : None | bool = None,
        constructAnalyticalSolution : None | bool = None
        ):
    
    problemParams = [
        [ # Problem 1
            [
                [0.0, 3.0, 1.0, 0.5, 100.0, 1.0, 7.0, 0.1,],
                {
                    "homogenousSolutionOnly" : False,
                    "constructAnalyticalSolution" : True,
                    "t0" : 0.0,
                    "ic" : [0.0, 0.0],
                }
            ]
        ],
        [ # Problem 2
            [
                [0.0, 3.0, 2.0, 0.8, 150.0, 2.0, 5.0, 0.05],
                {
                    "homogenousSolutionOnly" : False,
                    "constructAnalyticalSolution" : True,
                    "t0" : 0.0,
                    "ic" : [0.0, 0.0]
                }
            ]
        ],
        [ # Problem 3
            [
                [0.0, 3.0, 1.5, 0.4, 200.0, 1.5, 10.0, 0.01],
                {
                    "homogenousSolutionOnly" : False,
                    "constructAnalyticalSolution" : True,
                    "t0" : 0.0,
                    "ic" : [0.0, 0.0]
                }
            ]
        ],
        [ # Problem 4
            [
                [0.0, 10.0, 1.0, 0.0, 1.0, 0.0, 0.0, dt],
                {
                    "homogenousSolutionOnly" : True,
                    "constructAnalyticalSolution" : True,
                    "t0" : 0.0,
                    "ic" : [1.0, 1.0]
                }
            ] for dt in [0.1, 0.05, 0.01]
        ],
        [ # Problem 5
            [
                [0.0, 25.0, 1.0, 0.125, 1.0, 0.0, 0.0, dt],
                {
                    "homogenousSolutionOnly" : True,
                    "constructAnalyticalSolution" : True,
                    "t0" : 0.0,
                    "ic" : [2.0, 0.0]
                }
            ] for dt in [0.1, 0.05, 0.01]
        ],
        [ # Problem 6
            [
                [0.0, 70.0, 1.0, 0.125, 1.0, 3.0, 1.0, dt],
                {
                    "homogenousSolutionOnly" : False,
                    "constructAnalyticalSolution" : True,
                    "t0" : 0.0,
                    "ic" : [2.0, 0.0]
                }
            ] for dt in [0.1, 0.05, 0.01]
        ],
        [ # Problem 7
            [
                [0.0, 100.0, 1.0, 0.0, 1.0, 0.5, 0.8, dt],
                {
                    "homogenousSolutionOnly" : False,
                    "constructAnalyticalSolution" : True,
                    "t0" : 0.0,
                    "ic" : [1.0, 0.0]
                }
            ] for dt in [0.1, 0.05, 0.01]
        ]
    ]
    
    kwargs = {
        "homogenousSolutionOnly" : homogenousSolutionOnly,
        "constructAnalyticalSolution" : constructAnalyticalSolution
    }
    for subProblemParams in problemParams:
        for p in subProblemParams:
            for k, v in kwargs.items():
                if v is not None:
                    p[1][k] = v

    return problemParams
    
if __name__ == "__main__":
    main()