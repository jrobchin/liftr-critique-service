from celery import Celery

celery_app = Celery('critiques',
                    backend='redis://redis',
                    broker='redis://redis',
                    include=['critique.tasks'])