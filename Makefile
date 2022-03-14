.PHONY: all
all: install

# Installing project dependencies.
.PHONY: install
install: pipenv
	pipenv install
	sudo apt install -y libopenblas-dev

.PHONY: pipenv
pipenv:
	pip install pipenv

.PHONY: install-dev
install-dev: install
	pipenv install --dev

.PHONY: test
test:
	pipenv run pytest src/tests
