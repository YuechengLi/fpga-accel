#!/usr/bin/env python3
"""Stage 5 per-pass unit tests (unittest, stdlib only -- no pytest
on this VM). Run: python3 -m unittest discover -s tests -v
(or sw/run_stage5.sh).

Each test file covers one compiler pass. The tests use hand-built IR
graphs (fast, no torch) except test_capture.py, which exercises the
real torch.export frontend.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run_all(verbosity=1):
    loader = unittest.TestLoader()
    suite = loader.discover(os.path.dirname(os.path.abspath(__file__)),
                            pattern="test_*.py")
    runner = unittest.TextTestRunner(verbosity=verbosity)
    return runner.run(suite)


if __name__ == "__main__":
    res = run_all(verbosity=2 if "-v" in sys.argv else 1)
    sys.exit(0 if res.wasSuccessful() else 1)
