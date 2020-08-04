"""
functions and classes to support lpParse.py
"""

import constants
from collections import defaultdict

# from pulp import


class Matrix:
    """ output matrix class """

    class Objective:
        def __init__(self, expression, sense, name):
            if name:
                self.name = name[0]
            else:
                self.name = ""
            self.sense = sense  # 1 is minimise, -1 is maximise
            self.expression = expression
            # a dict with variable names as keys, and coefficients as values

    class Constraint:
        def __init__(self, expression, sense, rhs, name):
            if name:
                self.name = name[0]
            else:
                self.name = ""
            self.sense = sense  # 1 is geq, 0 is eq, -1 is leq
            self.rhs = rhs
            self.expression = expression

    class Variable:
        def __init__(self, bounds, category, name):
            self.name = name
            self.bounds = (bounds["lb"], bounds["ub"])  # a tuple (lb, ub)
            self.category = category  # 1 for int, 0 for linear

    def __init__(self, parserObjective, parserConstraints,
                 parserBounds, parserGenerals, parserBinaries):

        self.objective = Matrix.Objective(varExprToDict(
            parserObjective.varExpr),
            constants.objSenses[parserObjective.objSense], parserObjective.name)

        self.constraints = [Matrix.Constraint(varExprToDict(
            c.varExpr), constants.constraintSenses[c.sense], c.rhs, c.name)
            for c in parserConstraints]

        # can't get parser to generate this dict because one var can have several bound statements
        boundDict = getBoundDict(parserBounds, parserBinaries)

        allVarNames = set()
        allVarNames.update(self.objective.expression.keys())
        for c in self.constraints:
            allVarNames.update(c.expression.keys())
        allVarNames.update(parserGenerals)
        allVarNames.update(boundDict.keys())

        self.variables = [Matrix.Variable(boundDict[vName], ((vName in list(parserGenerals)) or (
            vName in list(parserBinaries))), vName) for vName in allVarNames]

    def __repr__(self):
        return "Objective%s\n\nConstraints (%d)%s\n\nVariables (%d)%s" \
            % ("\n%s %s %s" % (self.objective.sense, self.objective.name,
                               str(self.objective.expression)),
                len(self.constraints),
               "".join(["\n(%s, %s, %s, %s)" % (c.name, str(c.expression), c.sense, c.rhs)
                        for c in self.constraints]),
               len(self.variables),
               "".join(["\n(%s, %s, %s)" % (v.name, str(v.bounds), v.category)
                        for v in self.variables]))


def varExprToDict(varExpr):
    return dict((v.name[0], v.coef) for v in varExpr)


def getBoundDict(parserBounds, parserBinaries):
    # need this versatility because the lb and ub can come in separate bound statements
    boundDict = defaultdict(lambda: {"lb": -constants.infinity, "ub": constants.infinity})

    for b in parserBounds:
        bName = b.name[0]

        # if b.free, default is fine

        if b.leftbound:
            if constants.constraintSenses[b.leftbound.sense] >= 0:  # NUM >= var
                boundDict[bName]["ub"] = b.leftbound.numberOrInf

            if constants.constraintSenses[b.leftbound.sense] <= 0:  # NUM <= var
                boundDict[bName]["lb"] = b.leftbound.numberOrInf

        if b.rightbound:
            if constants.constraintSenses[b.rightbound.sense] >= 0:  # var >= NUM
                boundDict[bName]["lb"] = b.rightbound.numberOrInf

            if constants.constraintSenses[b.rightbound.sense] <= 0:  # var <= NUM
                boundDict[bName]["ub"] = b.rightbound.numberOrInf

    for bName in parserBinaries:
        boundDict[bName]["lb"] = 0
        boundDict[bName]["ub"] = 1

    return boundDict


def multiRemove(baseString, removables):
    """ replaces an iterable of strings in removables
        if removables is a string, each character is removed """
    for r in removables:
        try:
            baseString = baseString.replace(r, "")
        except TypeError:
            raise TypeError
    return baseString
