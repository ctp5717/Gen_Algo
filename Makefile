.PHONY: test

test:
	pre-commit run --all-files
	pytest -q -n auto --cov=. --cov-branch --cov-report=xml --cov-fail-under=80 --strict-markers --strict-config -W error::DeprecationWarning --junitxml=pytest-junit.xml --durations=10
