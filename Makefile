PYTHON ?= .venv/bin/python

.PHONY: check fast slow all lint type demo clean

check: fast lint type

# The fast tier: every test that does not require the slow (real process /
# long sleep) marker. Today only the unshare parametrization skips itself.
fast:
	$(PYTHON) -m pytest -m "not slow" --durations=10

slow:
	$(PYTHON) -m pytest -m slow --durations=10

all:
	$(PYTHON) -m pytest

lint:
	$(PYTHON) -m ruff check src tests

type:
	$(PYTHON) -m mypy src

demo:
	$(PYTHON) -m src.three_agent_demo

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache
