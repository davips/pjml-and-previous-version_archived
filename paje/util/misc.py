from functools import reduce


def flatten(l):
    """
    Flatten a list of lists removing duplicates.
    :param l:
    :return:
    """
    sets = [set(item) for item in l]
    result = set()
    for s in sets:
        result.update(s)
    return list(result)


def unsandwich(st):
    """
    Extract content between outermost parentheses.
    :param s: string
    :return:
    """
    return st[st.find("(") + 1:st.rfind(")")]
