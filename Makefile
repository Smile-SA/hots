all: venv install done

venv:
	python3.10 -m venv venv

.PHONY: install
install: venv
	./venv/bin/pip install --upgrade setuptools
	./venv/bin/pip install -U pip wheel
	./venv/bin/pip install -e .[dev,kafka]

.PHONY: doc
doc:
	./venv/bin/pip install -e .[doc]
	./venv/bin/pip install sphinx-rtd-theme
	./venv/bin/sphinx-build -b html doc/source doc/build
	
.PHONY: done
done:
	@ echo "Installation finished succesfully. \
	Run 'hots path/to/config/file' to start the application or \
	'hots --help' for command description"


.PHONY: clean
clean:
	rm -rf .pytest_cache .eggs *.egg-info
	find . -path ./venv -prune -o -name "*.pyc" -o -name "*.pyo" -o -name "__pycache__" -print0 | xargs -r -0 rm -rv
	@echo "You may not want to remove ./venv, please do it by hand." >&2
