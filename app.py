import os
from flask import Flask, render_template, request, redirect
from werkzeug.utils import secure_filename
from segmentation.kmeans import segment_leaf

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
RESULT_FOLDER = 'static/results'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER

# Pastikan folder ada
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# fungsi untuk membersihkan "static/" dari path kmeans
def clean_static_path(path):
    if path.startswith("static/"):
        return path.replace("static/", "", 1)
    return path

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'image' not in request.files:
            return redirect(request.url)

        file = request.files['image']
        k = int(request.form.get('k', 10))

        if file.filename == '':
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Jalankan segmentasi
            data = segment_leaf(filepath, k)

            # Path gambar utama
            original_rel = f"uploads/{filename}"
            result_rel = clean_static_path(data["result_path"])
            bar_rel = clean_static_path(data["bar_chart"])

            # Bersihkan path cluster
            for c in data["clusters"]:
                c["path"] = clean_static_path(c["path"])

            return render_template(
                'result.html',
                original=original_rel,
                result=result_rel,
                bar_chart=bar_rel,
                clusters=data["clusters"],
                numeric_data=data["numeric_data"],
                k=k
            )

    return render_template(
        'index.html',
        original=None,
        result=None,
        bar_chart=None,
        clusters=None,
        numeric_data=None,
        k=10
    )

if __name__ == '__main__':
    app.run(debug=True)
