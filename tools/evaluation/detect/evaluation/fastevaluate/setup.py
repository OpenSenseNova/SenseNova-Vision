import glob
from distutils.core import setup, Extension

sources = glob.glob("fastevaluate/*.cpp") + glob.glob("fastevaluate/*.c")

module = Extension('fastevaluate', 
                   language='c++',
                   sources=sources,
                   extra_compile_args=['-O3', '-lpthread'])

setup(name='fastevaluate',
    version='2.0',
    description='test',
    packages=['fastevaluate'],
    ext_modules=[module],
    scripts=['scripts/cocoap'],
)
