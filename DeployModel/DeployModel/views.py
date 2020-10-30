from django.http import HttpResponse
from django.shortcuts import render
from tensorflow.keras.models import load_model
import numpy as np


def home(request):
    new_model = load_model("keras-model.h5")

    lis = []
    lis.append(request.POST.get("inp1"))
    lis.append(request.POST.get("inp2"))
    lis.append(request.POST.get("inp3"))
    lis.append(request.POST.get("inp4"))
    lis.append(request.POST.get("inp5"))
    lis.append(request.POST.get("inp6"))
    lis.append(request.POST.get("inp7"))
    lis.append(request.POST.get("inp8"))
    lis.append(request.POST.get("inp9"))
    lis.append(request.POST.get("inp10"))
    lis.append(request.POST.get("inp11"))
    lis.append(request.POST.get("inp12"))
    lis.append(request.POST.get("inp13"))
    lis.append(request.POST.get("inp14"))
    lis.append(request.POST.get("inp15"))
    lis.append(request.POST.get("inp16"))
    lis.append(request.POST.get("inp17"))

    arr = np.array(lis, dtype=float).reshape(1, 17)
    inp = arr.tolist()

    out = []
    out.append(request.POST.get("guess"))
    output = np.array(out, dtype=float).reshape(1, 1)
    outt = output.tolist()

    scores = new_model.evaluate(inp, outt)
    if scores[1] * 100 == 100.0:
        ans = ">>> The User will Buy a Product 🛒"
    else:
        ans = "NO Revenue generated 🗋"

    return render(request, "home.html", {"ans": ans})


def notebook(request):
    return render(request, "notebook.html")


def report(request):
    return render(request, "report.html")