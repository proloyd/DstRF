import pytest

_Dict = {'True': True, 'False': False}

# decide on cross-validation test from command line
def pytest_addoption(parser):
    parser.addoption(
        "--crossvalidation", action="store", default="False", help="run crossvalidation? True or False"
    )


@pytest.fixture
def cmdopt(request):
    return _Dict.get(request.config.getoption("--crossvalidation"), None)
