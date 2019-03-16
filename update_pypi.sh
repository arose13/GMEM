#!/usr/bin/env bash
# Requires package `twine`
# Requires `.pypirc` file to be in the home directory. eg `~/.pypirc`
mv dist/*tar.gz old_dist/

python setup.py sdist
python3 -m twine upload dist/*

echo "--- DONE ---"