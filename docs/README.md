

## readme


To start building the docs we first need to install the software listed in 
the ipcoal/docs/environment.yml file, which is the file that tells readthedocs
the software it needs to build it.
```bash
conda install sphinx ipykernel nbsphinx sphinx_rtd_theme -c conda-forge
```

Then you can `cd` into the docs/ dir and initialize the directory for sphinx
by calling sphinx. 

```bash
sphinx-quickstart -p ipcoal -a "Patrick McKenzie & Deren Eaton" -v "0.0.2" -l "python"
```

This will create conf.py, index.rst, Makefile, and make.bat. To build the docs locally just call `make html` and then open the `_html/index.html` file in a web browser. 

Edits to the 'conf.py' file:
- set theme: 'html_theme = 'sphinx_rtd_theme'
- allow building from notebooks: 'extensions = ["nbsphinx"]'


