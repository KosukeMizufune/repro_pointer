from urllib.parse import urlparse
from pathlib import Path

import click

from repro_pointer import download


AVAILABLE_DATASETS = {
    'ModelNet40': {
        'url': 'https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip'  # noqa
    }
}


@click.command()
@click.argument('name', type=str)
@click.option('--data_root_dir', type=click.Path(), default="data")
@click.pass_context
def main(ctx, name, data_root_dir):
    data = AVAILABLE_DATASETS[name]
    url = data['url']
    zip_filename = Path(data_root_dir) / Path(urlparse(url).path).name
    if not zip_filename.exists():
        download.urlretrieve(url, zip_filename)
    dest_dir = zip_filename.with_suffix('')
    ext = zip_filename.suffix
    if not dest_dir.exists():
        download.extractall(zip_filename, dest_dir.parent, ext)


if __name__ == "__main__":
    main()
