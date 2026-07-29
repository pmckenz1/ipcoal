---
section: Getting Started
---

# Contributing

**Collaborator's are very welcome!**

If you haven’t already, you’ll want to first get familiar with the `ipcoal`
repository at [http://github.com/eaton-lab/ipcoal](http://github.com/eaton-lab/ipcoal). 
There you will find the source code and issue tracker where you can inquire
about ongoing development, discuss planned contributions, and volunteer to take
on known issues or future planned developments.

## Getting started
To contribute as a developer you'll need to install `ipcoal` from source
from GitHub and install additional dependencies used for testing the code.
Our workflow for this is to clone the repository (in your case, a fork of the
repo) and install in development mode using pip.

```bash
# install dependencies from conda
$ conda install ipcoal -c eaton-lab --only-deps

# clone the repo and cd into it
$ git clone https://github.com/eaton-lab/ipcoal.git
$ cd ipcoal/

# call pip install in 'development mode' (note the '-e .')
$ pip install -e . --no-deps
```
