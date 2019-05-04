from setuptools import setup, find_packages

setup(name="paje",
      version="0.0.1",
      packages=find_packages(),
      )

setup(name='paje',
      version='0.1',
      description='Paje automated machine leaning tool.',
      url='https://github.com/ealcobaca/automl-paje',
      author=["Edesio Alcobaça"],
      author_email='e.alcobaca@gmail.com',
      license='GPL3',
      packages=['paje'],
      install_requires=[
          "scipy",
          "catboost",
          "numpy",
          "scikit-learn",
          "imbalanced-learn",
          "liac-arff",
          "numpy",
          "pandas"
      ],
      setup_requires=["pytest-runner"],
      tests_require=["pytest"],
      zip_safe=False)
