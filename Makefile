all: clean

clean:
	find . -name "*.so" -o -name "*.pyc" -o -name "*.md5" -o -name "*.pyd" -o -name "*~" | xargs rm -f
	rm -rf .coverage.*
	rm -rf dist
	rm -rf build
	rm -rf docs/_build
	rm -rf docs/source/generated
	rm -rf docs/source/auto_examples
