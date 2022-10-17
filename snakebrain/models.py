from django.db import models
from django.contrib.postgres import fields

class SnakeState(models.Model):
	game_id = models.PositiveIntegerField(blank=True, null=True)
	order_id = models.PositiveIntegerField(blank=True, null=True)

	score = models.PositiveSmallIntegerField(blank=True, null=True)

	action = models.PositiveSmallIntegerField(blank=True, null=True)
	snake_dead = models.BooleanField(blank=True, null=True)
	snake_fed = models.BooleanField(blank=True, null=True)

	snakehead = models.PositiveSmallIntegerField(blank=True, null=True)

	snakebody = fields.ArrayField(
		models.PositiveSmallIntegerField(blank=True, null=True),
		blank=True, null=True
	)
	walls = fields.ArrayField(
		models.PositiveSmallIntegerField(blank=True, null=True),
		blank=True, null=True
	)
	apples = fields.ArrayField(
		models.PositiveSmallIntegerField(blank=True, null=True),
		blank=True, null=True
	)
	scissors = fields.ArrayField(
		models.PositiveSmallIntegerField(blank=True, null=True),
		blank=True, null=True
	)

	def __str__(self):
		return 'game: ' + str(self.game_id) + \
			' turn: ' + str(self.order_id)