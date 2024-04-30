"""Run the bench of hots with one dataset, specifying the parameters."""

import subprocess

import click

# TODO add 'help' message

datasets = [
    'profiles_stab_5n_50i',
    'profiles_stab_10n_217i',
    'profiles_add_5n_68i',
    'profiles_add_10n_477i',
    'profiles_change_5n_71i',
    'profiles_change_10n_197i',
    'profiles_del_5n_57i',
    'profiles_del_10n_261i'
]
# methods = ['init', 'spread', 'iter-consol', 'heur', 'loop']
methods = ['loop']
cluster_methods = ['loop-cluster',
                   'kmeans-scratch',
                   'stream-km']


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

    output_path = '../bench/global_eval/%s/' % path.split('/')[3]

    for k in range(kmin, kmax + 1, kstep):
        for tau in range(taumin, taumax + 1, taustep):
            for method in methods:
                if method == 'loop':
                    for cluster_method in cluster_methods:
                        temp_output = '%sk%d_tau%d_%s_%s' % (
                            output_path, k, tau, method, cluster_method
                        )
                        bash_command = '\
                            hots %s -k %d -t %d -m %s -c %s -o %s\
                        ' % (
                            path, k, tau, method, cluster_method, temp_output
                        )
                        print('\n%s\n' % bash_command)
                        process = subprocess.Popen(bash_command.split())
                        output, error = process.communicate()
                else:
                    temp_output = '%sk%d_tau%d_%s' % (
                        output_path, k, tau, method
                    )
                    bash_command = 'hots %s -k %d -t %d -m %s -o %s' % (
                        path, k, tau, method, temp_output
                    )
                    print('\n%s\n' % bash_command)
                    process = subprocess.Popen(bash_command.split())
                    output, error = process.communicate()


if __name__ == '__main__':
    main()
