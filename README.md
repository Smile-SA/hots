# **HOTS**

> Application for testing a hybrid resource allocation method using machine learning and optimization.

## Requirements for running HOTS

HOTS works on any platform with Python 3.8 and up.

The dev Python version must be used to install the package (for example install the package
python3.10-dev in order to use Python 3.10).

A solver needs to be installed for using HOTS. By default, GLPK is installed with HOTS, but the
user needs to install the following packages before using HOTS :
 * libglpk-dev
 * glpk-utils

## Installing HOTS

A Makefile is given, which creates a virtual environment and install HOTS. You can do :

```bash
make
```

## Running HOTS

```bash
hots /path/to/data/folder
```

Make sure to activate the virtual environment before running HOTS with :

```bash
source venv/bin/activate
```

You can see the help running :
```bash
hots --help
```

Note that a test data is given with the package, so you can easily test the installation with :
```bash
hots /tests/data/generated_7
```

## Credits

This software is sponsored by [Smile](https://www.smile.fr/).

The team:

- Jonathan Rivalan - Project manager
- Etienne Leclercq - Software design, lead developer
- Marco Mariani
- Gilles Lenfant

## Links

- [Project home](https://git.rnd.smile.fr/overboard/soft_clustering/rac)
- [File issues (bugs, ...)](https://git.rnd.smile.fr/overboard/soft_clustering/rac/-/issues)

## License

This software is provided under the terms of the MIT license you can read in the `LICENSE.txt` file of the repository or the package.
