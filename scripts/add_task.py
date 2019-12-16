import os

import click
from critique.celery import celery_app

@click.command()
@click.argument('path')
def add_task(path):
    abs_path = os.path.abspath(path)
    celery_app.send_task('critique.tasks.critique_image', args=[path])


if __name__ == "__main__":
    add_task() # pylint: disable=no-value-for-parameter