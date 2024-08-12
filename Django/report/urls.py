from django.urls import path

from . import views

# app_name = 'report'

urlpatterns = [
    path("", views.display_report, name="report"),
    path('send/', views.send_message, name='send_message'),
]