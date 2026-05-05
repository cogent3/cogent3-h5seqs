import os

import nox

_py_versions = range(10, 15)
# on python >= 3.12 this will improve speed of test coverage a lot
os.environ["COVERAGE_CORE"] = "sysmon"

nox.options.reuse_existing_virtualenvs = True
nox.options.default_venv_backend = "uv"


@nox.session(python=False)
def fmt(session: nox.Session) -> None:
    session.run("ruff", "check", "--fix-only", ".", external=True)
    session.run("ruff", "format", ".", external=True)


@nox.session(python=[f"3.{v}" for v in _py_versions])
def type_check(session):
    session.install("-e", ".", "--group", "dev")
    session.run("mypy", "src/cogent3_h5seqs")


@nox.session(python=[f"3.{v}" for v in _py_versions])
def test(session):
    session.install("-e", ".", "--group", "dev")
    # doctest modules within cogent3/app
    session.run(
        "pytest",
        "-s",
        "-x",
        ".",
        *session.posargs,
    )


@nox.session(python=[f"3.{v}" for v in _py_versions])
def testcov(session):
    session.install("-e", ".", "--group", "dev")
    # doctest modules within cogent3/app
    session.run(
        "pytest",
        "--cov-report",
        "html",
        "--cov",
        "cogent3_h5seqs",
        ".",
        *session.posargs,
    )
