from django.shortcuts import render
from django.http import HttpResponse
from django.core.exceptions import PermissionDenied
from .models import SnakeState

def control(request):
	pass

def movesnake(request):
	if request.method == 'POST':
		jsondata = request.read().decode()
		SnakeState(StateJSON=jsondata).save()

		return HttpResponse("Success");
	else:
		raise PermissionDenied