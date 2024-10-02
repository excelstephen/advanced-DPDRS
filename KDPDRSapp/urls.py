from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('signup/', views.signup, name='signup'),
    path('home/', views.home, name='home'),
    path('drug-recommendation/', views.drugRecommendation, name = 'drug-recommendation'),
    path('logout/', views.logout_view, name='logout')

]