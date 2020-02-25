import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="easymh",
    version="0.0.1",
    author="Wenjie Zheng",
    author_email="work@zhengwenjie.net",
    description="A lightweight Markov Chain Monte Carlo package with focus on Metropolis-Hastings.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/WenjieZ/easyMH",
    license='MIT',
    packages=setuptools.find_packages(),
    classifiers=['Intended Audience :: Science/Research',
                 'Intended Audience :: Developers',
                 'License :: OSI Approved',
                 'Programming Language :: Python',
                 'Topic :: Software Development',
                 'Topic :: Scientific/Engineering',
                 'Operating System :: Microsoft :: Windows',
                 'Operating System :: POSIX',
                 'Operating System :: Unix',
                 'Operating System :: MacOS',
                  ],
)
