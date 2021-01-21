
long_description = f"""
'Combine MUSE HII-regions with HST star clusters'
"""

from setuptools import setup, find_namespace_packages

setup(name='cluster',
      version='0.1',
      author='Fabian Scheuermann',
      author_email='f.scheuermann@uni-heidelberg.de',
      license='MIT',
      package_dir={"": "src"},
      packages=find_namespace_packages(where="src"),
      description='Combine MUSE HII-regions with HST star clusters',
      long_description = long_description)