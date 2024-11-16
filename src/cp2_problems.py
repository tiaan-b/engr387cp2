import matplotlib.pyplot as plt
import numpy as np

from src import lodecci

def main():
    plt.close('all')
    
    problems = getProblemSet()
    for i, problem in enumerate(problems[:3]):
        models, xs, ts, axs, x_analyticals = problem(showPlot = False)
        fig = axs[0].get_figure()
        fig.suptitle(f"Problem {i + 1}")
        fig.set_tight_layout(True)
        
    plt.show()

def getProblemSet():
    problemParams = getProblemParams()
    problems = []

    for subProblemParams in problemParams:
        subProblems = [
            getGenericProblem(*p[0], **p[1]) for p in subProblemParams
        ]
        problems.append(packageProblems(subProblems))
        
    return problems  
    
def getGenericProblem(tStart, tStop, *args, generalSolutionOnly: bool = True, **kwargs):
    def p(ax = None, showPlot = False):
        model = lodecci.MSDp_SinusoidalForcing_IVP_Model(*args, **kwargs)
        if generalSolutionOnly:
            model = model.getHomogenousModel()
            
        x, t, axOut, x_analytical = model.quickPlot(
            tStop,
            tStart = tStart,
            zerothOrderOnly = True,
            ax = ax,
            showPlot = showPlot,
            plotAnalyticalSolution = "analyticalSolution" in kwargs
        )
        
        return model, x, t, axOut, x_analytical
    
    return p

def packageProblems(problems):
    def p(axs = None, showPlot = False):
        if axs is None:
            _, axs = plt.subplots(
                ncols = len(problems), 
                squeeze = False, 
                sharey = True
            )
    
        out = []
        for problem, ax in zip(problems, axs.flatten()):
            out.append(problem(ax = ax, showPlot = showPlot))
        out = [list(x) for x in zip(*out)]

        return tuple(out)
    
    return p

def getProblemParams():
    return [
        [ # Problem 1
            [
                [0.0, 10.0, 1.0, 0.5, 100.0, 1.0, 7.0, 0.1,],
                {
                    "generalSolutionOnly" : False,
                    "t0" : 0.0,
                    "ic" : [0.0, 0.0],
                }
            ]
        ],
        [ # Problem 2
            [
                [0.0, 10.0, 2.0, 0.8, 150.0, 2.0, 5.0, 0.05],
                {
                    "generalSolutionOnly" : False,
                    "t0" : 0.0,
                    "ic" : [0.0, 0.0]
                }
            ]
        ],
        [ # Problem 3
            [
                [0.0, 10.0, 1.5, 0.4, 200.0, 1.5, 10.0, 0.01],
                {
                    "generalSolutionOnly" : False,
                    "t0" : 0.0,
                    "ic" : [0.0, 0.0]
                }
            ]
        ],
        [ # Problem 4
            [
                [0.0, 10.0, 1.0, 0.0, 1.0, 0.0, 0.0, dt],
                {
                    "generalSolutionOnly" : True,
                    "t0" : 0.0,
                    "ic" : [1.0, 1.0]
                }
            ] for dt in [0.1, 0.05, 0.01]
        ],
        [ # Problem 5
            [
                [0.0, 25.0, 1.0, 0.125, 1.0, 0.0, 0.0, dt],
                {
                    "generalSolutionOnly" : True,
                    "t0" : 0.0,
                    "ic" : [2.0, 0.0]
                }
            ] for dt in [0.1, 0.05, 0.01]
        ],
        [ # Problem 6
            [
                [0.0, 70.0, 1.0, 0.125, 1.0, 3.0, 1.0, dt],
                {
                    "generalSolutionOnly" : False,
                    "t0" : 0.0,
                    "ic" : [2.0, 0.0]
                }
            ] for dt in [0.1, 0.05, 0.01]
        ],
        [ # Problem 7
            [
                [0.0, 100.0, 1.0, 0.0, 1.0, 0.5, 0.8, dt],
                {
                    "generalSolutionOnly" : False,
                    "t0" : 0.0,
                    "ic" : [1.0, 0.0]
                }
            ] for dt in [0.1, 0.05, 0.01]
        ]
    ]
    
if __name__ == "__main__":
    main()