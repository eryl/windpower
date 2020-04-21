# coding: utf-8

from setuptools import setup

def readme():
    with open('README.txt') as f:
        return f.read()

setup(name='windpower',
      version='0.1',
      description='Windpower forecasting using the Greenlytics weather API',
      url='https://github.com/eryl/windpower',
      author='Erik Ylipää',
      author_email='erik.ylipaa@ri.se',
      license='MIT',
      packages=['windpower'],
      install_requires=['mltrain', 'numpy', 'xarray', 'matplotlib', 'pandas', 'tqdm', 'scikit-learn', 'requests'],
      dependency_links=['http://github.com/eryl/mltrain'],
      zip_safe=False)
