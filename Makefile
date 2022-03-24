.PHONY: all
all: install

.PHONY: pipenv
pipenv:
	pip install pipenv

.PHONY: install
install: pipenv
	pipenv install
	sudo apt install -y libopenblas-dev

.PHONY: install-dev
install-dev: pipenv
	pipenv install --dev

.PHONY: test
test:
	pipenv run pytest src/tests
