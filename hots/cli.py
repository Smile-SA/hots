import click
from config.loader import load_config
from core.app import App

@click.command()
@click.option('--config', '-c', 'config_path', required=True, type=click.Path(exists=True), help='Path to JSON config file')
def main(config_path):
    cfg = load_config(config_path)
    app = App(cfg)
    app.run()

if __name__ == '__main__':
    main()