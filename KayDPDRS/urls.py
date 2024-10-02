from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('KDPDRSapp.urls')),
    path('', include('django.contrib.auth.urls')),  # For built-in authentication views like logout
] 
