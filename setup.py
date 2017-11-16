from setuptools import setup

setup(name='hydromet_forecasting',
      version='0.1',
      description='Forecasting 5-day, decadal, monthly discharge',
      url='https://github.com/julesair/hydromet-forecasting',
      author='Jules Henze',
      author_email='henze@hydrosolutions.ch',
      license='MIT',
      packages=['hydromet_forecasting'],
      install_requires=[
          'enum',
          'numpy',
          'datetime',
          'pandas',
          'sklearn',
      ],
      zip_safe=False)
