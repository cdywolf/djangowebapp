from django.urls import path
from . import views
from .views import *

urlpatterns = [
    path('', views.prediction_view, name='prediction_view'),
    path('send_email/', views.send_email, name='send_email'),
]
