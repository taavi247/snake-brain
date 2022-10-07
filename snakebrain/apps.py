from django.apps import AppConfig
from snakebrain import globalvariables

class SnakebrainConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'snakebrain'

    def ready(self):
        globalvariables.snakenetwork.start_snakenetwork()