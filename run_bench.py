"""
=========
run bench cots
=========
Run the bench of cots with one dataset, specifying the parameters k and tau you want.
"""

import subprocess

import click

# TODO add 'help' message

# methods = ['init', 'spread', 'iter-consol', 'heur', 'loop']
methods = ['init', 'spread', 'iter-consol', 'heur']
# methods = ['loop']
cluster_methods = ['loop-cluster',
                   'kmeans-scratch']


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
    taustep = taustep or 5

    output_path = '../bench/global_eval/AlibabaV1_50n/'

    # tol_clust_min = 4
    # tol_clust_max = 4
    # tol_place_min = 4
    # tol_place_max = 4

    for k in range(kmin, kmax + 1, kstep):
        for tau in range(taumin, taumax + 1, taustep):
            for method in methods:
                if method == 'loop':
                    for cluster_method in cluster_methods:
                        temp_output = '%sk%d_tau%d_%s_%s' % (
                            output_path, k, tau, method, cluster_method
                        )
                        bash_command = 'cots %s -k %d -t %d -m %s -c %s -o %s' % (
                            path, k, tau, method, cluster_method, temp_output
                        )
                        print('\n%s\n' % bash_command)
                        process = subprocess.Popen(bash_command.split())
                        output, error = process.communicate()
                else:
                    temp_output = '%sk%d_tau%d_%s' % (
                        output_path, k, tau, method
                    )
                    bash_command = 'cots %s -k %d -t %d -m %s -o %s' % (
                        path, k, tau, method, temp_output
                    )
                    print('\n%s\n' % bash_command)
                    process = subprocess.Popen(bash_command.split())
                    output, error = process.communicate()

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

    # for tol_c in range(tol_clust_min, tol_clust_max):
    #     epsilon_c = tol_c / 10
    #     for tol_a in range(tol_place_min, tol_place_max):
    #         epsilon_a = tol_a / 10
    #         temp_output = output_path + str(tol_c) + '_' + str(tol_a) + '/'
    #         bash_command = 'cots ' + path + ' -o ' + temp_output + \
    #             ' -ec ' + str(epsilon_c) + ' -ea ' + str(epsilon_a)
    #         print('\n%s\n' % bash_command)
    #         process = subprocess.Popen(bash_command.split())
    #         output, error = process.communicate()


if __name__ == '__main__':
    main()
