def perform(args):
    x = args[0]
    return x, shape_function(args)
def shape_function(x):
    return np.sin(x[0])+x[1]
if __name__ == "__main__":
    import numpy as np
    print perform((1,0,3))
