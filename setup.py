from setuptools import setup
from os import path


HERE = path.abspath(path.dirname(__file__))


def readme():
    with open(path.join(HERE, 'README.rst')) as f:
        return f.read()


setup(
    name='choix',
    version='0.3.3',
    author='Lucas Maystre',
    author_email='lucas@maystre.ch',
    description="Inference algorithms for models based on Luce's choice axiom.",
    long_description=readme(),
    url='https://github.com/lucasmaystre/choix',
    license='MIT',
    classifiers=[
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    keywords='statistics ml bradley terry plackett luce choice comparison ranking',
    packages=['choix'],
    install_requires=[
        'numpy',
        'scipy',
    ],
    setup_requires=['pytest-runner'],
    tests_require=[
        'pytest',
        'networkx',
    ],
    include_package_data=True,
    zip_safe=False,
)
