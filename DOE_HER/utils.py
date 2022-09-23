import numpy as np


def finite_difference(derivative_order=1, window=5):
    """Generate finite difference coefficients.

    Args:
        derivative_order (int, optional): Order to calculate the derivative to. Defaults to 1.
        window (int, optional): Centered window of points to calculate over. Defaults to 5.

    Returns:
        np.array[float]: Finite difference coefficients
    """
    assert window % 2, "Window must be odd"
    assert derivative_order == 1, (
        "derivative order %i not implemented yet" % derivative_order
    )
    fin_diff_lookup = [
        [
            [-0.5, 0, 0.5],
            [1 / 12, -2 / 3, 0, 2 / 3, -1 / 12],
            [-1 / 60, 3 / 20, -3 / 4, 0, 3 / 4, -3 / 20, 1 / 60],
            [1 / 280, -4 / 105, 1 / 5, -4 / 5, 0, 4 / 5, -1 / 5, 4 / 105, -1 / 280],
        ]
    ]

    return np.array(fin_diff_lookup[derivative_order - 1][int(window / 2) - 1])
