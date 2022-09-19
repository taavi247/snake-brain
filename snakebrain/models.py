from django.db import models

class SnakeState(models.Model):
	 snakeLocation = models.CharField(max_length=900)
	 itemLocation = models.CharField(max_length=900)