# **rac**

Script for testing a hybrid resource allocation method using machine learning and optimization.

### _Installation_

```bash
# Install dependencies
python setup.py install
make
```

### _Environment_

```bash
# Activate the virtual environment
. venv/bin/activate
```

### _Usage_

```bash
rac data_folder
```

### _Data_

At the moment, the data folder must contain the following files :

- container_usage.csv : describes the containers resource consumption ;
- node_usage.csv : describes the nodes resource consumption ;
- node_meta.csv : describes the nodes capacities ;

The repository gives two example datasets in _data_ folder (_generated_10_ and _generated_30_).
