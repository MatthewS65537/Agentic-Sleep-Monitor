from django.urls import path

from . import views

# app_name = 'report'

urlpatterns = [
    path("", views.display_report, name="report"),
]