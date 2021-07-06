"""
=========
run bench cots
=========
Run the bench of cots with one dataset, specifying the parameters k and tau you want.
"""

import subprocess

data_path = '/home/eleclercq/Documents/CIFRE/data/alter_way/15-22_03_21/'

k_min = 2
k_max = 10
k_step = 1

tau_min = 25
tau_max = 300
tau_step = 50


def main():
    """Perform the bench."""
    for k in range(k_min, k_max, k_step):
        for tau in range(tau_min, tau_max, tau_step):
            bash_command = 'cots --path ' + data_path + ' --k ' + str(k) + ' --tau ' + str(tau)
            process = subprocess.Popen(bash_command.split())
            output, error = process.communicate()


if __name__ == '__main__':
    main()
