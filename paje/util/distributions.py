from math import floor


def exponential_integers(kmax, only_odd=True, exponent=1.5):
    """
    Generates a list of odd ks increasing exponentially according to x^1.5: [1, 3, 5, 9, 11, 15, 19, ..., 37, 43, 47, 53, ..., 83, 89, 97, 103, ..., 955, 971, 985, ...]
    :param kmax: maximum allowed number
    :param only_odd: odd numbers only
    :param exponent: level of increase
    :return: list of numbers
    """
    ks = []
    for x in list(range(1, floor(pow(kmax, 1 / exponent)))):
        k = round(pow(x, exponent))
        ks.append(k + 1 if only_odd and k % 2 == 0 else k)
    return ks