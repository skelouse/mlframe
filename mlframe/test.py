import importlib
import pandas as pd
import warnings
import matplotlib as mpl
import mlframe
importlib.reload(mlframe)
from mlframe import MLFrame # noqa
mpl.use('pdf')
warnings.filterwarnings('ignore')

df_test = None


class Tester():
    failed = 0

    def passed_deco(func):
        """Wrapper for __call__ to assert and count
        failures of tests"""
        def wrapper(self, test_func, e, *args, **kwargs):
            global df_test
            # to keep df loaded, doesn't have to be reloaded for each test
            df = df_test.copy()
            df.model = None
            try:
                print('Checking', test_func.__name__, end='--')
                a = test_func.main(df)
                try:
                    assert(a == e)
                except ValueError:  # needed to assert arrays
                    assert((a == e).all())
            except AssertionError:
                self.failed += 1
                print('FAILED! #%s' % self.failed)
                if test_func.__name__ == "tests.fail_test":
                    print("as expected..")
                else:
                    print('#'*30)
            else:
                print('passed')
            return func
        return wrapper

    @passed_deco
    def __call__(self):
        """Override of call so Tester()(x) can be used
        i.e calling of an instance of Tester"""
        pass

    def run_tests(self):
        """Run all known tests"""
        print('\n\n\nStart Test\n')
        fail_test(self)
        clean_col_names(self)
        get_vif(self)
        get_vif_cols(self)
        log(self)
        scale(self)
        one_hot_encode(self)
        find_outliers_IQR(self)
        find_outliers_Z(self)
        get_nulls(self)
        drop_nulls_perc(self)
        ms_matrix(self)

        # deprecated
        # fill_na_mode(self)
        # fill_na_mean(self)

        qq_plot(self)
        model_resid_scatter(self)
        lrmodel(self)
        plot_corr(self)
        plot_coef(self)
        regplot(self)
        boxplot(self)
        get_cols(self)
        outlier_removal(self)
        find_outliers_cooks_d(self)
        distplot(self)
        jointplot(self)

        if self.failed == 1:
            print("Failed 1 test, as expected")
        elif self.failed == 0:
            print("Fail test failed, error in Tester")
        else:
            print("Failed %s tests" % self.failed)


def fail_test(t):  # added
    from tests import fail_test
    t(fail_test, 1)


def clean_col_names(t):  # added
    from tests import clean_col_names
    expected = ['mpg', 'cylinders', 'displacement', 'horsepower',
                'weight', 'acceleration', 'model_year', 'origin', 'car_name']
    t(clean_col_names, expected)


def get_vif(t):  # added
    from tests import get_vif
    expected = [763.5575, 10.7375, 21.8368, 9.9437,
                10.8313, 2.6258, 1.245, 1.7724]
    t(get_vif, expected)


def get_vif_cols(t):  # added
    from tests import get_vif_cols
    expected = ['horsepower', 'cylinders', 'weight', 'displacement']
    t(get_vif_cols, expected)


def log(t):  # added
    from tests import log
    expected = [2.8904, 2.7081, 2.8904, 2.7726, 2.8332, 2.7081,
                2.6391, 2.6391, 2.6391, 2.7081, 2.7081, 2.6391,
                2.7081, 2.6391, 3.1781, 3.091, 2.8904, 3.0445,
                3.2958, 3.2581]
    t(log, expected)


def scale(t):  # added
    from tests import scale
    expected = [-0.6986, -1.0835, -0.6986, -0.9552, -0.8269, -1.0835,
                -1.2118, -1.2118, -1.2118, -1.0835, -1.0835, -1.2118,
                -1.0835, -1.2118, 0.0711, -0.1855, -0.6986, -0.3138,
                0.4559, 0.3277]
    t(scale, expected)


def one_hot_encode(t):  # added
    from tests import one_hot_encode
    expected = 6  # sum of model_volvo
    t(one_hot_encode, expected)


def find_outliers_IQR(t):  # added
    from tests import find_outliers_IQR
    expected = 10  # sum of horsepower IQR outliers
    t(find_outliers_IQR, expected)


def find_outliers_Z(t):  # added
    from tests import find_outliers_Z
    expected = 5  # sum of horsepower z_score outliers
    t(find_outliers_Z, expected)


def outlier_removal(t):  # added
    from tests import outlier_removal
    expected = 10
    t(outlier_removal, expected)


def get_nulls(t):  # added
    from tests import get_nulls
    expected = 2
    t(get_nulls, expected)


def drop_nulls_perc(t):  # added
    from tests import drop_nulls_perc
    expected = 1
    t(drop_nulls_perc, expected)


def ms_matrix(t):  # added
    from tests import ms_matrix
    expected = 1
    t(ms_matrix, expected)


# deprecated
"""
def fill_na_mode(t):  # added
    from tests import fill_na_mode
    expected = 0
    t(fill_na_mode, expected)


def fill_na_mean(t):  # added
    from tests import fill_na_mean
    expected = 0
    t(fill_na_mean, expected)
"""


def qq_plot(t):  # added
    from tests import qq_plot
    expected = 1
    t(qq_plot, expected)


def model_resid_scatter(t):  # added
    from tests import model_resid_scatter
    expected = 1
    t(model_resid_scatter, expected)


def lrmodel(t):  # added
    from tests import lrmodel
    expected = 0.96
    t(lrmodel, expected)


def plot_corr(t):  # added
    from tests import plot_corr
    expected = 1
    t(plot_corr, expected)


def plot_coef(t):  # added
    from tests import plot_coef
    expected = 1
    t(plot_coef, expected)


def regplot(t):  # added
    from tests import regplot
    expected = 1
    t(regplot, expected)


def distplot(t):  # added
    from tests import distplot
    expected = 1
    t(distplot, expected)


def jointplot(t):  # added
    from tests import jointplot
    expected = 1
    t(jointplot, expected)


def boxplot(t):  # added
    from tests import boxplot
    expected = 1
    t(boxplot, expected)


def get_cols(t):  # added
    from tests import get_cols
    expected = ['mpg', 'cylinders', 'displacement', 'acceleration', 'car name']
    t(get_cols, expected)


def train_test_split(t):  # added
    from tests import train_test_split
    expected = 0.5028
    t(train_test_split, expected)


def fill_na_kind(t):  # added
    from tests import fill_na_kind
    expected = {'mode': {0: 150.0, 1: 150.0, 2: 150.0},
                'mean': {0: 104.1311, 1: 104.1311, 2: 104.1311},
                'median': {0: 92.0, 1: 92.0, 2: 92.0},
                'perc': 0,
                'custom': 0}
    t(fill_na_kind, expected)


def find_outliers_cooks_d(t):
    from tests import find_outliers_cooks_d
    expected = 21
    t(find_outliers_cooks_d, expected)


# skeleton
"""
def xxxxx(t):
    from tests import xxxxx
    expected =
    t(xxxxx, expected)
"""


def quick_test(dft):
    """For building new tests"""
    from tests import find_outliers_cooks_d
    importlib.reload(find_outliers_cooks_d)
    from tests import find_outliers_cooks_d
    print(find_outliers_cooks_d.main(df_test))


def test_all(df):
    """Called by load_then_test.py for hot_swapping code
    with the df staying the same
    """
    t = Tester()
    global df_test
    df_test = MLFrame(df)

    # change full to 0 for quick test
    full = 1
    if full:
        t.run_tests()
    else:
        quick_test(df_test)
    # from tests import lr_model
    # expected = [0.9364463356461579, 0.9492392408557554, 0.9720009970902538,
    #             0.8200601719712063, 0.8927759998561855, 0.8902204648630595,
    #             0.861693791723534, 0.9270817920824502, 0.8776814581431275,
    #             0.8761528624812863]
    # assert((test(lr_model.main).pvalues[0:10] == expected).all())


if __name__ == "__main__":
    df = MLFrame(pd.read_csv('mlframe/tests/auto-mpg.csv'))
    test_all(df)
