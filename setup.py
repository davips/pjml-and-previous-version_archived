"""Setup for pjml package."""
import setuptools

import pjml

NAME = "pjml"

VERSION = pjml.__version__

AUTHOR = 'Edesio and Davi Pereira dos Santos'

AUTHOR_EMAIL = ''

DESCRIPTION = 'Machine learning library'

with open('README.md', 'r') as fh:
    LONG_DESCRIPTION = fh.read()

LICENSE = 'GPL3'

URL = 'https://github.com/automated-data-science/paje-ml'

DOWNLOAD_URL = 'https://github.com/automated-data-science/paje-ml/releases'

CLASSIFIERS = ['Intended Audience :: Science/Research',
               'Intended Audience :: Developers',
               'License :: OSI Approved :: GNU General Public License v3 ('
               'GPLv3)',
               'Natural Language :: English',
               'Programming Language :: Python',
               'Topic :: Software Development',
               'Topic :: Scientific/Engineering',
               'Operating System :: OS Independent',
               'Programming Language :: Python :: 3.6',
               'Programming Language :: Python :: 3.7']

INSTALL_REQUIRES = [
    'imblearn', 'methodtools', 'pymfe', 'sklearn', 'compose',
    #'cururu @ git+https://github.com/davips/cururu@master#egg=package1.0',
    'numpy',  # required by tests.py    
]

EXTRAS_REQUIRE = {
    'code-check': [
        'pylint',
        'mypy'
    ],
    'tests': [
        'pytest',
        'pytest-cov',
    ],
    'docs': [
        'sphinx',
        'sphinx-gallery',
        'sphinx_rtd_theme',
        'numpydoc'
    ]
}

SETUP_REQUIRES = ['flake8', 'autopep8', 'wheel']

setuptools.setup(
    name=NAME,
    version=VERSION,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    license=LICENSE,
    url=URL,
    download_url=DOWNLOAD_URL,
    packages=setuptools.find_packages(),
    classifiers=CLASSIFIERS,
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
    setup_requires=SETUP_REQUIRES    
)

package_dir = {'': 'pjml'}  # For IDEs like Intellij to recognize the package.
