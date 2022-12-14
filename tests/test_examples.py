#!/usr/bin/env python3
# encoding: utf-8
"""
tests.test_examples
~~~~~~~~~~~~~~~~~~~

Run standalone python files that are a complete examples.
Used to test the full examples in the documentation.

:author: Wannes Meert
:copyright: Copyright 2015-2022 DTAI, KU Leuven and Sirris.
:license: Apache License, Version 2.0, see LICENSE for details.
"""
import sys
import os
import logging
from pathlib import Path
import subprocess as sp
import pytest
import leuvenmapmatching as mm


logger = mm.logger
examples_path = Path(os.path.realpath(__file__)).parent / "examples"


def test_examples():
    example_fns = examples_path.glob("*.py")
    for example_fn in example_fns:
        execute_file(example_fn.name)


def importrun_file(fn, cmp_with_previous=False):
    import importlib
    fn = f"examples.{fn[:-3]}"
    print(f"Importing: {fn}")
    o = importlib.import_module(fn)
    o.run()


def execute_file(fn, cmp_with_previous=False):
    print(f"Testing: {fn}")
    fn = examples_path / fn
    assert fn.exists()
    try:
        cmd = sp.run(["python3", fn], capture_output=True, check=True)
    except sp.CalledProcessError as exc:
        print(exc)
        print(exc.stderr.decode())
        print(exc.stdout.decode())
        raise exc

    if cmp_with_previous:
        # Not ready to be used in general testing, output contains floats
        result_data = cmd.stdout.decode()
        correct_fn = fn.with_suffix(".log")
        if correct_fn.exists():
            with correct_fn.open("r") as correct_fp:
                correct_data = correct_fp.read()
            print(correct_data)
            print(result_data)
            assert correct_data == result_data, f"Logged output different for {fn}"
        else:
            with correct_fn.open("w") as correct_fp:
                correct_fp.write(result_data)


if __name__ == "__main__":
    logger.setLevel(logging.WARNING)
    logger.addHandler(logging.StreamHandler(sys.stdout))
    directory = Path(os.environ.get('TESTDIR', Path(__file__).parent))
    print(f"Saving files to {directory}")
    # execute_file("example_1_simple.py", cmp_with_previous=True)
    execute_file("example_using_osmnx_and_geopandas.py", cmp_with_previous=True)
    # importrun_file("example_using_osmnx_and_geopandas.py", cmp_with_previous=True)
