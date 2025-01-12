from django.shortcuts import render, redirect, HttpResponse
from django.contrib.auth.models import User
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.contrib.auth import authenticate, login
from django.views.decorators.csrf import csrf_exempt, csrf_protect
from django.contrib.auth import logout
import requests
import random
import json
from django.http import JsonResponse
import simplejson
from django.contrib.auth.tokens import default_token_generator
from django.utils.http import urlsafe_base64_encode, urlsafe_base64_decode
from django.utils.encoding import force_bytes, force_str
from django.core.mail import send_mail
import tensorflow as tf
import os
import pandas as pd
import numpy as np
import h5py
import keras

print(os.getcwd())
file_path = os.path.abspath("app/loan_percentage_model.h5")
loan_model = tf.keras.models.load_model(file_path)
file_path1 = os.path.abspath("app/interest_rate_model.keras")
interest_model = tf.keras.models.load_model(file_path1)
file_path2 = os.path.abspath("app/riskscore.h5")
risk_model = tf.keras.models.load_model(file_path2)

# Create your views here.
def about_us(request):
    return render(request, 'main/about_us.html')

def sign_up(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        email = request.POST.get('email')
        pass1 = request.POST.get('password1')
        pass2 = request.POST.get('password2')

        if pass1 == pass2:
            user = User.objects.create_user(username, email, pass1)
            user.save()
            
            return redirect('login')
        else:
            messages.error(request, "Passwords do not match")
    return render(request, 'main/signup.html')

def get_interest_rate(request):
    s=42
    
    random.seed(s + int(request))
    
    interest_rate = random.uniform(0, 4)
    return int(interest_rate)

def get_user_info(request):

    return render(request, 'main/get_user_info.html')

def submit_info(request):
    if request.method == 'POST':
        age = request.POST.get('Age')
        annual_income = request.POST.get('AnnualIncome')
        credit_score = request.POST.get('CreditScore')
        experience = request.POST.get('Experience')
        loan_duration = request.POST.get('LoanDuration')
        loan_amount = request.POST.get('LoanAmount')
        dependents = request.POST.get('NumberOfDependents')
        monthly_debt = request.POST.get('MonthlyDebtPayments')
        credit_utilization = request.POST.get('CreditCardUtilizationRate')
        open_credit_lines = request.POST.get('NumberOfOpenCreditLines')
        credit_inquiries = request.POST.get('NumberOfCreditInquiries')
        debt_to_income = request.POST.get('DebtToIncomeRatio')
        bankruptcy = request.POST.get('BankruptcyHistory')
        previous_defaults = request.POST.get('PreviousLoanDefaults')
        savings_balance = request.POST.get('SavingsAccountBalance')
        checking_balance = request.POST.get('CheckingAccountBalance')
        total_assets = request.POST.get('TotalAssets')
        total_liabilities = request.POST.get('TotalLiabilities')
        monthly_income = request.POST.get('MonthlyIncome')
        net_worth = request.POST.get('NetWorth')
        total_debt_to_income = request.POST.get('TotalDebtToIncomeRatio')
        risk_score = request.POST.get('RiskScore')
        bank_approved = request.POST.get('Approved')
        bank_interest = request.POST.get('InterestRate')
        print(bankruptcy)
        input_data = {
            "Age": age,
            "AnnualIncome": annual_income,
            "CreditScore": credit_score,
            "Experience": experience, 
            'LoanAmount': loan_amount,
            "LoanDuration": loan_duration,
            "NumberOfDependents": dependents,
            "MonthlyDebtPayments": monthly_debt, 
            "CreditCardUtilizationRate": credit_utilization,  # Add corresponding form field
            "NumberOfOpenCreditLines": open_credit_lines,  # Add corresponding form field
            "NumberOfCreditInquiries": credit_inquiries,  # Add corresponding form field
            "DebtToIncomeRatio": debt_to_income,  # Add corresponding form field
            "BankruptcyHistory": bankruptcy,  # Add corresponding form field
            "PreviousLoanDefaults": previous_defaults,  # Add corresponding form field
            "SavingsAccountBalance": savings_balance,  # Add corresponding form field
            "CheckingAccountBalance": checking_balance,  # Add corresponding form field
            "TotalAssets": total_assets,  # Add corresponding form field
            "TotalLiabilities": total_liabilities,  # Add corresponding form field
            "MonthlyIncome": monthly_income,  # Add corresponding form field
            "NetWorth": net_worth,  # Add corresponding form field
            "TotalDebtToIncomeRatio": total_debt_to_income, 
            "RiskScore": risk_score,
            }

        encoded_input = pd.DataFrame([input_data])
        #print(encoded_input.values)
        
        model_input = encoded_input.values
        model_input = encoded_input.to_numpy(dtype="float32")
        #print(model_input)
        prediction = loan_model.predict(model_input)
        predicted_approval_probability = prediction[0][0]
        prediction1 = interest_model.predict(model_input)
        predicted_interest_model = prediction1[0][0]
        print(predicted_interest_model)
        predicted_risk = max(int(risk_score) - get_interest_rate(int(loan_amount)+1), 1)
        print(predicted_approval_probability)
        print(predicted_interest_model)
        
        context = {
            "predicted_approval_probability": predicted_approval_probability,
            "bank_approved": bank_approved,
            "predicted_interest_rate": predicted_interest_model,
            "bank_interest": bank_interest,
            "predicted_risk_score": predicted_risk,
            "bank_risk_score": risk_score,
        }

        return render(request, 'main/compare.html', context)
    
    return redirect('user_info')

@csrf_exempt
def login_own(request):
    if request.method == "POST":
        username = request.POST.get('username')
        password = request.POST.get('password1') 
        user = authenticate(request, username=username, password=password)
        if user is not None:
            login(request, user)
            return redirect('submit_info')
        else:
            error_message = "Invalid username or password"
            return render(request, 'main/login.html', {'error_message': error_message})
    return render(request, 'main/login.html')

def logout_user(request):
    logout(request)
    return redirect('sign_up')

@csrf_exempt
def reset_check_email(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        email = data.get('email')
        print(data, email)
        if User.objects.filter(email=email).exists():
            user = User.objects.get(email=email)
            token = default_token_generator.make_token(user)
            uid = urlsafe_base64_encode(force_bytes(user.pk))
            reset_link = f"http://localhost:8000/reset_user_password/{uid}/{token}/"
            s = f'Please follow this link to reset your password: {reset_link}'
            send_mail(
                    'Change Password for CCC',
                    f'{s}',
                    'rt.scheduling.automailer@gmail.com',
                    [email],
                    fail_silently=False,
                )
            return JsonResponse({'exists': True})
        else:
            return JsonResponse({'exists': False})
            
    return render(request, 'main/password_reset.html')

def reset_user_password(request, uid, token):
    if request.method == 'POST':
        uid = force_str(urlsafe_base64_decode(uid))
        user = User.objects.get(pk=uid)
        password = request.POST.get('password')
        if user is not None and default_token_generator.check_token(user, token):
            user.set_password(password)
            print(user.password)
            user.save()
            return redirect('login')

    return render(request, 'main/password_reset_form.html')

def password_reset_sent(request):
    return render(request, 'main/password_reset_sent.html')

def donate1(request):
    return render(request, 'main/donate.html')