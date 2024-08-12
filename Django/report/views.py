from django.shortcuts import render, redirect
from .models import Message
# Create your views here.
from django.http import HttpResponse

def display_report(request):
    messages = Message.objects.all()
    return render(request, 'report/report.html', {'messages': messages})

def send_message(request):
    print(request.POST)
    if request.method == 'POST':
        username = request.POST.get('username')
        content = request.POST.get('content')
        if username and content:
            Message.objects.create(username=username, content=content)
    return redirect('report')