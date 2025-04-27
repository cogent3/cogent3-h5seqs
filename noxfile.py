import nox

_py_versions = range(10, 14)


@nox.session(python=[f"3.{v}" for v in _py_versions])
def test(session):
    session.install("-e.[dev]")
    # doctest modules within cogent3/app
    session.run(
        "pytest",
        "-s",
        "-x",
        ".",
    )
