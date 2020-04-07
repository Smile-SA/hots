# **rac**

> Application for testing a hybrid resource allocation method using machine learning and optimization.

(**TODO**: describe a **real life** use case for `rac`)

## Installation

### Common requirements

Have Python 3.6, 3.7 or 3.8 on an Unix box (Linux, MacOS, ...) with a graphic display.

### Production

Latest stable version:

```bash
# Install dependencies
python setup.py install
make
```

See `pip` documentation and available versions on PyPI for other options and `rac` versions.

### Development

We assume you activated a dedicated virtual environment with Python 3.6, 3.7 or 3.8 with whatever
tool you prefer (virtualenv, venv, pew, pyenv, ...), and you cloned `rac` from its Git repository.

```bash
# Activate the virtual environment
. venv/bin/activate
```

The `dev` option adds development / tests tools.

### Full documentation

Start as in above **Development** section, then issue the commands:

```bash
pip install -e .[doc]
python setup.py build_sphinx
```

You can now open `build/sphinx/index.html` with your favorite Web browser.

## Usage

Display the short instructions with this command:

```bash
rac --help
```

## Credits

This software is sponsored by [Alter Way](https://www.alterway.fr/).

The team:

- Jonathan Rivalan - Project manager
- Etienne Leclercq - Software design, lead developer
- Marco Mariani
- Gilles Lenfant

## Links

- [Project home](https://git.rnd.alterway.fr/overboard/soft_clustering/rac)
- [File issues (bugs, ...)](https://git.rnd.alterway.fr/overboard/soft_clustering/rac/-/issues)

## License

This software is provided under the terms of the MIT license you can read in the `LICENSE.txt` file
of the repository or the package.
