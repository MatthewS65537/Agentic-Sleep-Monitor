from django.urls import path
from . import views

urlpatterns = [
    path('', views.dataview, name='dataview'),
    path('api/get-json-data/', views.get_json_data, name='get_json_data'),
]