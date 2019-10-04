from setuptools import setup

setup(name='hydromet_forecasting',
      version='0.3.15',
      description='Forecasting 5-day, decadal, monthly and seasonal discharge',
      url='https://github.com/julesair/hydromet-forecasting',
      author='Jules Henze',
      author_email='henze@hydrosolutions.ch',
      license='MIT',
      packages=['hydromet_forecasting'],
      install_requires=[
          'enum34>=1.1.6',
          'numpy>=1.14.5',
          'python-dateutil>=2.7.3',
          'pandas>=0.23.1',
          'sklearn>=0.0',
          'scipy>=1.1.0',
          'matplotlib>=2.2.2',
          'stldecompose>=0.0.3',
          'monthdelta>=0.9.1'
      ],
      zip_safe=False,
      include_package_data=True)
