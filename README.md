# **cots**

> Application for testing a hybrid resource allocation method using machine learning and optimization.

(**TODO**: describe a **real life** use case for `cots`)

## Installation

### Common requirements

Have Python 3.6, 3.7 or 3.8 on an Unix box (Linux, MacOS, ...) with a graphic display.

Moreover, `cots` module needs `IBM CPLEX` solver to be installed on the machine to
solve the optimization problem. To see details, visit their website [here](https://www.ibm.com/uk-en/products/ilog-cplex-optimization-studio).
Once you have `CPLEX` installed on the machine, you have to add in your
`#PYTHONPATH` the `/path/to/cplex/python/[python-version]/[your-distribution]`.

### Production

Latest stable version:

```bash
pip install cots
```

See `pip` documentation and available versions on PyPI for other options and `cots` versions.

### Development

The easiest way to install a `development` version of `cots` is to use `make` :

```bash
make
```

As it is created within the `Makefile`, we recommend using a dedicated virtual environment.
After the install has finished succesfully, don't forget to activate this environment :

```bash
source venv/bin/activate
```

The `dev` option adds development / tests tools.

### Full documentation

As in above **Development** section, the documentation can be built using `make`:

```bash
make doc
```

You can now open `build/sphinx/index.html` with your favorite Web browser.

## Usage

Basically, start the application with this command :

```bash
cots /path/to/your/data/folder
```

Note that the repository provides some examples datasets in `.tests/data/` folder.
You can then run, for instance, the `generated_30` example like this :

```bash
cots tests/data/generated_30
```

Your data folder must provide at least two files :

- `container_usage.csv` provides resource consumption by containers
- `node_meta.csv` provides nodes capacities for each metric

Display the short instructions with this command :

```bash
cots --help
```

## Known issues

- If you have any error (_ really any ? _) with matplotlib, you might have to install the backend
  `tkinter` with :

```bash
sudo apt install python3-tk
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
