

## readme

To start building the docs we first need to install the software listed in 
the ipcoal/docs/environment.yml file, which is the file that tells readthedocs
the software it needs to build it.
```bash
conda install sphinx ipykernel nbsphinx sphinx_rtd_theme -c conda-forge
```


### Testing changes to the docs

Since the docs have already been built previously you can test and view new
changes to the docs by running the following. Please test changes before committing them. The docs will be built in HTML format into the `_html` which 
you can then view by opening the index.html file in a browser. The docs include
some relative paths so it important that you `cd` into the docs dir to build.

```bash
cd docs/
sphinx . _html/
firefox _html/index.html
```


#### The first time building the docs 

**The first time** building the docs you `cd` into the docs/ dir and initialize the directory for sphinx by calling the sphinx-quickstart command:

```bash
sphinx-quickstart -p ipcoal -a "Patrick McKenzie & Deren Eaton" -v "0.0.2" -l "python"
```

This createed conf.py, index.rst, Makefile, and make.bat. To build the docs locally just call `make html` and then open the `_html/index.html` file in a web browser. 

Edits to the 'conf.py' file:
- set theme: 'html_theme = 'sphinx_rtd_theme'
- allow building from notebooks: 'extensions = ["nbsphinx"]'


