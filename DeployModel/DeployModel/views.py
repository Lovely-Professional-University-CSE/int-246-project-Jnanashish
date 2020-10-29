
from django.http import HttpResponse
from django.shortcuts import render

# from tensorflow import keras

# model = keras.models.load_model('keras-model.h5')

def home(request):
    ans = 5
    return render(request, "home.html",{'ans':ans})


def notebook(request):
    return render(request, "notebook.html")


def report(request):
    return render(request, "report.html")