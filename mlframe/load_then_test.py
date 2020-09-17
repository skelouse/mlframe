from mlframe import MLFrame
import pandas as pd
import importlib
from traceback import format_exc

# def test(func):
#     df = main(pd.read_csv('mlframe/tests/auto-mpg.csv'))
#     return func.main(df)


def main(test, df):
    test(df)


if __name__ == "__main__":
    df = MLFrame(pd.read_csv('mlframe/tests/auto-mpg.csv'))
    df = df.copy()
    while True:
        try:
            import test
            importlib.reload(test)
            import test
            from test import test_all
            main(test_all, df)
        except Exception:
            print(format_exc())
        input('press enter to run all tests >')
