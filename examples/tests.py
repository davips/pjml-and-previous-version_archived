"""Test"""
from pjml.tool.data.modeling.supervised.classifier.dt import DT


def printable_test():
    """toy test."""
    dt_tree = DT()
    dt_tree.disable_pretty_printing()
    print(repr(dt_tree))
    print(dt_tree)
    dt_tree.enable_pretty_printing()
    print()
    print(repr(dt_tree))
    print(dt_tree)


def main():
    """Main function"""
    printable_test()


if __name__ == '__main__':
    main()
