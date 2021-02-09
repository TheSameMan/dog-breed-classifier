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

    cls, file, pic = '', '', ''
    if request.method == 'POST':
        req = request.files['file']
        if req:
            file_path = os.path.join('static/img/', req.filename)
            req.save(file_path)
            cls = model.predict(file_path)
            file = req.filename
            pic = file_path

    return render_template('predict.html', cls=cls, file=file, pic=pic)


if __name__ == '__main__':
    app.run(debug=True)
    del model
    del app
