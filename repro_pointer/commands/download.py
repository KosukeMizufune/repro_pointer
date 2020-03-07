from urllib.parse import urlparse
from pathlib import Path

import click

from repro_pointer.commands.dataset import DATA_DIR
from repro_pointer import download


modelnet40_url = 'https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip'


AVAILABLE_DATASETS = {
    'ModelNet40': {
        'url': modelnet40_url,
        'dir': DATA_DIR
    }
}


@click.command()
@click.argument('name', type=str)
@click.pass_context
def main(ctx, name):
    data = AVAILABLE_DATASETS[name]
    url = data['url']
    data_dir = data['dir']
    zip_filename = data_dir / Path(urlparse(url).path).name
    if not zip_filename.exists():
        download.urlretrieve(url, zip_filename)
    dest_dir = zip_filename.with_suffix('')
    ext = zip_filename.suffix
    if not dest_dir.exists():
        download.extractall(zip_filename, dest_dir.parent, ext)


if __name__ == "__main__":
    main()
