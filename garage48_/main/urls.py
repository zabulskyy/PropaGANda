from django.contrib import admin
from django.urls import path, include
from .views import *

urlpatterns = [
    path('', index),
    path('api_v_1', Api_v_1.as_view())

]
