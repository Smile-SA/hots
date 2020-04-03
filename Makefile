all: venv install done

venv:
	python3.8 -m venv venv

.PHONY: install
install: venv
	./venv/bin/pip install pip==20.0.2 wheel==0.32.3
	./venv/bin/pip install --no-deps -r requirements.txt
	./venv/bin/pip install --no-deps -e .

.PHONY: done
done:
	@ echo "Installation finished succesfully. Run 'rac /path/to/data/folder' to start the application"

# README.html: README.adoc
# 	asciidoctor -n README.adoc

# .PHONY: test
# test:
# 	sudo venv/bin/python setup.py test

.PHONY: clean
clean:
	rm -rf .pytest_cache .eggs
	find . -path ./venv -prune -o -name "*.pyc" -o -name "*.pyo" -o -name "__pycache__" -print0 | xargs -r -0 rm -rv
	@echo "You may not want to remove ./venv, please do it by hand." >&2