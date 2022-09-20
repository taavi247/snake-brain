from django.db import models

class SnakeState(models.Model):
		StateJSON = models.JSONField(blank=True, null=True)