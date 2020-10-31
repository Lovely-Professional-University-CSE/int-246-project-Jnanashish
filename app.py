from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import numpy as np




app = Flask(__name__)

@app.route('/')
def hello():
     return render_template("home.html")

@app.route('/predict', methods=['POST'])
def result():
    
    new_model = load_model("keras-model.h5")

    lis = []
    lis.append(request.form["inp1"])
    lis.append(request.form["inp2"])
    lis.append(request.form["inp3"])
    lis.append(request.form["inp4"])
    lis.append(request.form["inp5"])
    lis.append(request.form["inp6"])
    lis.append(request.form["inp7"])
    lis.append(request.form["inp8"])
    lis.append(request.form["inp9"])
    lis.append(request.form["inp10"])
    lis.append(request.form["inp11"])
    lis.append(request.form["inp12"])
    lis.append(request.form["inp13"])
    lis.append(request.form["inp14"])
    lis.append(request.form["inp15"])
    lis.append(request.form["inp16"])
    lis.append(request.form["inp17"])

    arr = np.array(lis, dtype=float).reshape(1, 17)
    inp = arr.tolist()

    out = []
    out.append(request.form["guess"])
    output = np.array(out, dtype=float).reshape(1, 1)
    outt = output.tolist()

    scores = new_model.evaluate(inp, outt)
    if scores[1] * 100 == 100.0:
        ans = ">>> The User will Buy a Product 🛒"
    else:
        ans = "NO Revenue generated 🗋"    
    
    return render_template("home.html", data = ans)

if __name__ == '__main__':
    app.run(debug=True)