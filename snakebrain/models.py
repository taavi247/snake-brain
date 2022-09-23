from django.db import models
from django.contrib.postgres import fields

class SnakeState(models.Model):
	gameid = models.PositiveIntegerField(blank=True, null=True)
	orderid = models.PositiveIntegerField(blank=True, null=True)

	snakehead = models.PositiveIntegerField(blank=True, null=True)

	snakebody = fields.ArrayField(
		models.PositiveIntegerField(blank=True, null=True),
		blank=True, null=True
	)
	apples = fields.ArrayField(
		models.PositiveIntegerField(blank=True, null=True),
		blank=True, null=True
	)
	scissors = fields.ArrayField(
		models.PositiveIntegerField(blank=True, null=True),
		blank=True, null=True
	)

	def __str__(self):
		return "game: " + str(self.gameid) + \
			" turn: " + str(self.orderid)