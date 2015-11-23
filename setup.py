from setuptools import setup

def readme():
    with open('README.md') as f:
        return f.read()

setup(name='choix',
      version='0.1',
      description='A library for research on choice models.',
      long_description=readme(),
      classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 2.7',
        'Topic :: Scientific/Engineering :: Mathematics',
      ],
      keywords='statistics ml bradley terry luce thurstone choice ranking',
      url='http://lucas.maystre.ch/',
      author='Lucas Maystre',
      author_email='lucas@maystre.ch',
      license='MIT',
      packages=['choix'],
      install_requires=[
          'numpy',
          'scipy',
      ],
      test_suite='nose.collector',
      tests_require=['nose'],
      include_package_data=True,
      zip_safe=False)
