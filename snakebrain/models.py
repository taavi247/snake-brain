from django.db import models
from django.contrib.postgres import fields

class SnakeState(models.Model):
	game_id = models.PositiveIntegerField(blank=True, null=True)
	order_id = models.PositiveIntegerField(blank=True, null=True)

	score = models.PositiveIntegerField(blank=True, null=True)

	action = models.PositiveIntegerField(blank=True, null=True)
	done = models.BooleanField(blank=True, null=True)

	snakehead = models.PositiveIntegerField(blank=True, null=True)

	snakebody = fields.ArrayField(
		models.PositiveIntegerField(blank=True, null=True),
		blank=True, null=True
	)
	walls = fields.ArrayField(
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
		return 'game: ' + str(self.game_id) + \
			' turn: ' + str(self.order_id)