# **rac**

Script for testing a hybrid resource allocation method using machine learning and optimization.

## Installation

### Common requirements

Have Python 3.6, 3.7 or 3.8 on an Unix.

### Production

Latest stable version:

```bash
pip install rac
```

See `pip`documentation and available versions on PyPI for other options and `rac` versions.

### Development

We assume you activated a dedicated virtual environment with Python 3.6, 3.7 or 3.8 with whatever
tool you prefer (venv, pew, pyenv, ...), and you cloned `rac`from its Git repository.

```bash
cd /where/you/cloned/rac
pip install -e .[dev]
```

The `dev` option adds development / tests tools.

## _Usage_

```bash
rac data_folder
```

### _Data_

At the moment, the data folder must contain the following files :

- container_usage.csv : describes the containers resource consumption ;
- node_usage.csv : describes the nodes resource consumption ;
- node_meta.csv : describes the nodes capacities ;

The repository gives two example datasets in _data_ folder (_generated_10_ and _generated_30_).
