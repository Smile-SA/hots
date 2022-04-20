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
@click.option('--kmin', required=False, type=int)
@click.option('--kmax', required=False, type=int)
@click.option('--taumin', required=False, type=int)
@click.option('--taumax', required=False, type=int)
@click.option('--kstep', required=False, type=int)
@click.option('--taustep', required=False, type=int)
def main(path, kmin, kmax, taumin, taumax, kstep, taustep):
    """Perform the bench."""
    kstep = kstep or 1
    taustep = taustep or 1

    output_path = '../bench/eval_epsilons/random_10n_300i/'

    tol_clust_min = 1
    tol_clust_max = 9
    tol_place_min = 1
    tol_place_max = 9

    # for k in range(kmin, kmax + 1, kstep):
    #     for tau in range(taumin, taumax + 1, taustep):
    #         bash_command = 'cots ' + path + ' -k ' + str(k) + ' -t ' + str(tau)
    #         print('\n%s\n' % bash_command)
    #         process = subprocess.Popen(bash_command.split())
    #         output, error = process.communicate()

    # bash_command =
    # 'cots ' + path + ' -k ' + str(k) + ' -t ' + str(tau) + ' -m loop_v2'
    # print('\n%s\n' % bash_command)
    # process = subprocess.Popen(bash_command.split())
    # output, error = process.communicate()

    # bash_command =
    # 'cots ' + path + ' -k ' + str(k) + ' -t ' + str(tau) + ' -m loop_kmeans'
    # print('\n%s\n' % bash_command)
    # process = subprocess.Popen(bash_command.split())
    # output, error = process.communicate()

    for tol_c in range(tol_clust_min, tol_clust_max):
        epsilon_c = tol_c / 10
        for tol_a in range(tol_place_min, tol_place_max):
            epsilon_a = tol_a / 10
            temp_output = output_path + str(tol_c) + '_' + str(tol_a) + '/'
            bash_command = 'cots ' + path + ' -o ' + temp_output + \
                ' -ec ' + str(epsilon_c) + ' -ea ' + str(epsilon_a)
            print('\n%s\n' % bash_command)
            process = subprocess.Popen(bash_command.split())
            output, error = process.communicate()


if __name__ == '__main__':
    main()
