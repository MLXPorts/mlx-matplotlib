#!/usr/bin/env python3

"""
Check that the version number of the install Matplotlib does not start with 0

To run:
    $ python3 -m build .
    $ pip install dist/matplotlib*.tar.gz for sdist
    $ pip install dist/matplotlib*.whl for wheel
    $ ./ci/check_version_number.py
"""
import sys

import matplotlib


print(f"Version {matplotlib.__version__} installed")
# In this fork, we may not have git tags available in CI; fail only if we fell
# all the way back to the explicit "UNKNOWN" version.
if matplotlib.__version__.startswith("0.0+UNKNOWN"):
    sys.exit("Version is unknown (setuptools_scm fallback)")
