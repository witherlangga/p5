import os
from flask import Flask, render_template, request, redirect
from werkzeug.utils import secure_filename
from segmentation.kmeans import segment_leaf

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':

        if 'image' not in request.files:
            return redirect(request.url)

        file = request.files['image']

        if file.filename == '':
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # === Proses segmentasi ===
            data = segment_leaf(filepath)

            return render_template(
                'result.html',
                original=filepath,
                result=data["result_path"],
                clusters=data["clusters"]   # ✅ BENAR
            )

    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
