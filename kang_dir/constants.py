MINIMIZE = 1
MAXIMIZE = -1

objSenses = {"max": MAXIMIZE, "maximum": MAXIMIZE, "maximize": MAXIMIZE,
             "min": MINIMIZE, "minimum": MINIMIZE, "minimize": MINIMIZE}

GEQ = 1
EQ = 0
LEQ = -1

constraintSenses = {"<": LEQ, "<=": LEQ, "=<": LEQ,
                    "=": EQ,
                    ">": GEQ, ">=": GEQ, "=>": GEQ}

infinity = 1E30
