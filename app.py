"Application for classifying dog breeds (133 classes)"
import os
from flask import Flask, request
from flask.templating import render_template
from model import DogClassifier

model = DogClassifier()
app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def predict():
    """classification endpoint"""

    if request.method == 'POST':
        req = request.files['file']
        file_path = None
        if req:
            file_path = os.path.join('static/img/', req.filename)
            req.save(file_path)

            return render_template('predict.html', cls=model(file_path),
                                   pic=file_path, file=req.filename,
                                   error=model.error)

    return render_template('predict.html')


if __name__ == '__main__':
    app.run(debug=True)
