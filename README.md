# **HOTS**

> Hybrid Optimization for Time Series  
> HOTS solves problems presented as time series thanks to machine learning and optimization methods.  
> The library supports multiple resource related problems (placement, allocation), presented as one or more metrics.

## Requirements for running HOTS

HOTS works on any platform with Python 3.8 and up.

The dev Python version must be used to install the package (for example install the package
python3.10-dev in order to use Python 3.10).

A solver needs to be installed for using HOTS. By default, GLPK is installed with HOTS, but the
user needs to install the following packages before using HOTS :
 * libglpk-dev
 * glpk-utils

## Installing HOTS

A Makefile is provided, which creates a virtual environment and install HOTS. You can do :

```bash
make
```

## Running HOTS

The application can be used simply by running :

```bash
hots /path/to/data/folder
```

Make sure to activate the virtual environment before running HOTS with :

```bash
source venv/bin/activate
```

Some parameters can be defined with the `hots` command, such as :
 * `k` : the number of clusters used in clustering ;
 * `tau` : the window size during the loop process ;
 * `param` : a specific parameter file to use.

All the CLI options are found running the `help` option :
```bash
hots --help
```

More parameters can be defined through a `.JSON` file, for which an example is provided in the `tests` folder. See the documentation, section `User manual`, for more details about all the parameters.  

Note that a test data is provided within the package, so you can easily test the installation with :
```bash
hots /tests/data/generated_7
```

## Credits

Authors:

- Etienne Leclercq - Software design, lead developer
- Jonathan Rivalan - Product owner, Lead designer 
- Marco Mariani
- Gilles Lenfant
- Soobash Daiboo
- Kang Du
- Amaury Sauret
- SMILE R&D

## Links

- [Project home](https://git.rnd.smile.fr/overboard/soft_clustering/rac)
- [File issues (bugs, ...)](https://git.rnd.smile.fr/overboard/soft_clustering/rac/-/issues)

## License

This software is provided under the terms of the MIT license you can read in the `LICENSE.txt` file of the repository or the package.
