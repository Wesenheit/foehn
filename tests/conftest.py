import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--nprocs",
        action="store",
        default="10",
        help="Number of MPI processes to spawn for isolated tests",
    )
    parser.addoption(
        "--nprocs-gpu",
        action="store",
        default="1",
        help="Number of MPI processes to spawn for isolated tests (GPU)",
    )


@pytest.fixture
def nprocs(request):
    return request.config.getoption("--nprocs")


@pytest.fixture
def nprocs_gpu(request):
    return request.config.getoption("--nprocs-gpu")
