"""
=========
run bench cots
=========
Run the bench of cots with one dataset, specifying the parameters k and tau you want.
"""

import subprocess

import click

# TODO add 'help' message


@click.command()
@click.option('--path', required=True, type=click.Path(exists=True))
@click.option('--kmin', required=True, type=int)
@click.option('--kmax', required=True, type=int)
@click.option('--taumin', required=True, type=int)
@click.option('--taumax', required=True, type=int)
@click.option('--kstep', required=False, type=int)
@click.option('--taustep', required=False, type=int)
def main(path, kmin, kmax, taumin, taumax, kstep, taustep):
    """Perform the bench."""
    kstep = kstep or 1
    taustep = taustep or 1

    for k in range(kmin, kmax + 1, kstep):
        for tau in range(taumin, taumax + 1, taustep):
            bash_command = 'cots --path ' + path + ' --k ' + str(k) + ' --tau ' + str(tau)
            print('\n%s\n' % bash_command)
            process = subprocess.Popen(bash_command.split())
            output, error = process.communicate()


if __name__ == '__main__':
    main()
