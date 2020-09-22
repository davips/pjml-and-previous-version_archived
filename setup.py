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
]

CLASSIFIERS = ['Intended Audience :: Science/Research',
               'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
               'Natural Language :: English',
               'Programming Language :: Python',
               'Topic :: Scientific/Engineering',
               'Operating System :: Linux',
               'Programming Language :: Python :: 3.8']


INSTALL_REQUIRES = [
    'numpy', 'sklearn', 'liac-arff'
]


EXTRAS_REQUIRE = {
}

SETUP_REQUIRES = ['wheel']

setuptools.setup(
    name=NAME,
    version=VERSION,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    classifiers=CLASSIFIERS,
    description=DESCRIPTION,
    download_url=DOWNLOAD_URL,
    extras_require=EXTRAS_REQUIRE,
    install_requires=INSTALL_REQUIRES,
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    license=LICENSE,
    packages=setuptools.find_packages(),
    setup_requires=SETUP_REQUIRES,
    url=URL,
)

package_dir = {'': 'pjml'}  # For IDEs like Intellij to recognize the package.
