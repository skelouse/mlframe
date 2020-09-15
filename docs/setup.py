from setuptools import setup

setup(name='mlframe',
      version='0.1.5',
      description="mlframe",
      long_description="",
      author='Sam Stoltenberg',
      author_email='sam@skelouse.com',
      license='GNU',
      packages=['mlframe'],
      zip_safe=False,
      install_requires=[
    'missingno>=0.4.2',
    'pandas>=1.1.1',
    'numpy>=1.19.1',
    'matplotlib>=3.3.1',
    'seaborn>=0.10.1',
    'statsmodels',
    'scipy>=1.3.1',
    'scikit-learn'
],
      )