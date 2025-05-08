import os

import nox

_py_versions = range(10, 14)
# on python >= 3.12 this will improve speed of test coverage a lot
os.environ["COVERAGE_CORE"] = "sysmon"


@nox.session(python=[f"3.{v}" for v in _py_versions])
def test(session):
    session.install("-e.[dev]")
    # doctest modules within cogent3/app
    session.run(
        "pytest",
        "-s",
        "-x",
        ".",
        *session.posargs,
    )


@nox.session(python=[f"3.{v}" for v in _py_versions])
def test_cov(session):
    session.install("-e.[dev]")
    # doctest modules within cogent3/app
    session.run(
        "pytest",
        "--cov-report",
        "html",
        "--cov",
        "cogent3_h5seqs",
        ".",
    )
