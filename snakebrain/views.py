from django.shortcuts import render
from django.http import HttpResponse
from django.core.exceptions import PermissionDenied
from .models import SnakeState
import json

def control(request):
	pass

def movesnake(request):
	if request.method == 'POST':
		jsondata = request.read().decode()
		jsondata_loaded = json.loads(jsondata, parse_int=int)
		print(jsondata_loaded['snakeBody'])
		SnakeState(
			gameid = jsondata_loaded['gameID'],
			orderid = jsondata_loaded['orderID'],
			snakehead = jsondata_loaded['snakeHead'],
			snakebody = jsondata_loaded['snakeBody'],
			apples = jsondata_loaded['apples'],
			scissors = jsondata_loaded['scissors'],
		).save()

		return HttpResponse("Success");
	else:
		raise PermissionDenied