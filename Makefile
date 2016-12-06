
PREFIX=/home/finsberg/local
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
	# python -m pytest -v tests/test_spatial_matparams.py
	python -m pytest -v tests/test_run.py

test_adjoint:
	python -m pytest -v tests/test_adjoint_calculations.py
