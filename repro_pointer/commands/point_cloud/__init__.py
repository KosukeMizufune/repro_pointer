import click

from repro_pointer.commands.point_cloud import download


@click.group(invoke_without_command=True)
@click.pass_context
def main(ctx, **kwargs):
    pass


main.add_command(download.main, 'download')
