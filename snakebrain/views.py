from django.shortcuts import render

def control(request):
	pass

def readstate(request):
	if request.POST:
		snakeLocation = request.POST['snakeLocation']
		itemLocation = request.POST['itemLocation']
		snakeState = SnakeState(snakeLocation, itemLocation)
		snakeState.save()
	else:
		raise PermissionDenied