import pytest

# decide on cross-validation test from command line
def pytest_addoption(parser):
    parser.addoption(
        "--crossvalidation", action="store", default="False", help="run crossvalidation? True or False"
    )


@pytest.fixture
def cmdopt(request):
    return request.config.getoption("--crossvalidation")
