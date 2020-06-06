def quadratic_eqn(x, a, b, c):
    '''
    Standard quadratic function we will pass into scipy's curve_fit function to approximate the balls trajectory
    once lost by the yolo model. We will use this to solve for y

    y = ax^2 + bx = c
    '''

    return (a * (x * x)) + (b * x) + c
