from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy as np


def numpy_include():
    try:
        numpy_include = np.get_include()
    except AttributeError:
        numpy_include = np.get_numpy_include()
    return numpy_include


ext_modules = [
    Extension(
        "eval_metrics_cy",
        ["eval_metrics_cy.pyx"],
        include_dirs=[numpy_include()],
    )
]

setup(name="Cython-based reid evaluation code", ext_modules=cythonize(ext_modules))
