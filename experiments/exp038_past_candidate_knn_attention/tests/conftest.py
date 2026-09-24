import pytest


@pytest.fixture(scope="session", autouse=True)
def bound_cpu_test_threads():
    try:
        import torch
    except ImportError:
        yield
        return

    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)
