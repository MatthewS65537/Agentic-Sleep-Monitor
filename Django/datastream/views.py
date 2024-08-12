from django.shortcuts import render

# Create your views here.
from django.http import HttpResponse

def datastream(request):
    # return HttpResponse("Hello, world. You're at the datastream index.")
    return render(request, 'datastream/datastream.html')