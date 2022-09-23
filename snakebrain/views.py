from django.shortcuts import render
from django.http import HttpResponse
from django.core.exceptions import PermissionDenied
from .models import SnakeState
from enum import Enum
import random
import json

class SnakeDirection(Enum):
	ArrowUp = 0
	ArrowDown = 1
	ArrowLeft = 2
	ArrowRight = 3

def movesnake(request):
	if request.method == 'POST':
		jsondata = request.read().decode()
		jsondata_loaded = json.loads(jsondata, parse_int=int)
		
		SnakeState(
			gameid = jsondata_loaded['gameID'],
			orderid = jsondata_loaded['orderID'],
			snakehead = jsondata_loaded['snakeHead'],
			snakebody = jsondata_loaded['snakeBody'],
			apples = jsondata_loaded['apples'],
			scissors = jsondata_loaded['scissors'],
		).save()

		snakeDirection = SnakeDirection(random.randint(0, 3)).name

		jsonresponse = json.dumps({ "snakeDirection": snakeDirection })
		
		return HttpResponse(
			jsonresponse, 
			content_type='application/json');
	else:
		raise PermissionDenied