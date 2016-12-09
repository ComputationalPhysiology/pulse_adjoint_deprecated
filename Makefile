
PREFIX=/usit/abel/u1/henriknf/mypackages
DOC=./doc/

default: install docs

install:
	python setup.py install --prefix=$(PREFIX)

docs:
	cd $(DOC); \
	make html; \
	cd -;

clean:
	clean-files; \
	rm -rf build;

test:
	python -m pytest -v tests/test_run.py

