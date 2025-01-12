from django.urls import path
from . import views

urlpatterns = [
    path('about-us/', views.about_us, name='about_us'),
    path('user-info/', views.get_user_info, name='user_info'),
    path('submit-info/', views.submit_info, name='submit_info'),
    path('', views.sign_up, name = 'sign_up'),
    path('login/', views.login_own, name = 'login'),
    path('logout/', views.logout_user, name = 'logout'),
    path('reset_check_email/', views.reset_check_email, name='reset_check_email'),
    path('password_reset_sent/', views.password_reset_sent, name='password_reset_sent'),
    path('reset_user_password/<uid>/<token>/', views.reset_user_password, name='reset_user_password'),
    path('donate/', views.donate1, name='donate'),
]