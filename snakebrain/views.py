from django.shortcuts import render
from django.http import HttpResponse
from django.core.exceptions import PermissionDenied
from .models import SnakeState
from snakebrain import globalvariables
from enum import Enum
import random
import json
import numpy as np

class SnakeDirection(Enum):
	ArrowUp = 0
	ArrowDown = 1
	ArrowLeft = 2
	ArrowRight = 3

def movesnake(request):
	if request.method == 'POST':
		jsondata = request.read().decode()
		jsondata_loaded = json.loads(jsondata, parse_int=int)

		if (
			jsondata_loaded.get('gameId') is None
			or jsondata_loaded.get('orderId') is None
			or jsondata_loaded.get('score') is None
			or jsondata_loaded.get('isSnakeDead') is None
			or jsondata_loaded.get('snakeHead') is None
			or jsondata_loaded.get('snakeBody') is None
			or jsondata_loaded.get('walls') is None
			or jsondata_loaded.get('apples') is None
			or jsondata_loaded.get('scissors') is None
		):
			print('POST did not contain correct JSON format ')

			return HttpResponse(
			'POST did not contain correct JSON format ' \
			'gameId, orderId, score, snakeHead, snakeBody, ' \
			'apples, scissors',
			content_type='text/html'
		);

		current_state = np.zeros(30 * 30)
		current_state[jsondata_loaded['apples']] = 1
		current_state[jsondata_loaded['scissors']] = 2
		current_state[jsondata_loaded['walls']] = 3
		current_state[jsondata_loaded['snakeHead']] = 4
		current_state[jsondata_loaded['snakeBody']] = 5

		snake_action = globalvariables.snakenetwork.get_action(current_state)
		snake_direction = SnakeDirection(snake_action).name

		SnakeState(
			game_id = jsondata_loaded['gameId'],
			order_id = jsondata_loaded['orderId'],
			score = jsondata_loaded['score'],
			action = snake_action,
			done = jsondata_loaded['isSnakeDead'],
			snakehead = jsondata_loaded['snakeHead'],
			snakebody = jsondata_loaded['snakeBody'],
			walls = jsondata_loaded['walls'],
			apples = jsondata_loaded['apples'],
			scissors = jsondata_loaded['scissors'],
		).save()

		json_response = json.dumps({ 'snakeDirection': snake_direction })

		return HttpResponse(
			json_response, 
			content_type='application/json'
		);
	elif request.method == 'GET':
		return HttpResponse(
			'This is the API to post Feed the Snake state ' \
			'to the server and get control response',
			content_type='text/html'
		);
	else:
		raise PermissionDenied