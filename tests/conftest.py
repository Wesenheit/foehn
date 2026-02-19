import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--nprocs",
        action="store",
        default="2",
        help="Number of MPI processes to spawn for isolated tests",
    )


@pytest.fixture
def nprocs(request):
    return request.config.getoption("--nprocs")
