from modules.config import app
from modules.color_extractor import color_extraction
from flask import render_template, request
import matplotlib.pyplot as plt
import numpy as np

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/extract', methods=["GET", 'POST'])
def extract():
    if request.method == 'POST':
        f = request.files['image'] # image is name in <input type="file" id="upload" name="image" required="true" class="me-3">
        k = request.form['k']
        k = int(k)
        filename = f.filename
        path = f'./static/uploads/{filename}'
        f.save(path)
        colors = color_extraction(filename, k)

        fig_width = max(1.2 * k, 2)  # keep width proportional to k but not too small
        plt.figure(figsize=(fig_width, 1.6), dpi=120)

        for i, color in enumerate(colors):
            plt.subplot(1, k, i+1)
            mat = np.zeros((100, 100, 3), dtype='uint8')
            mat[:,:,:] = color
            plt.imshow(mat)
            plt.title(f'RGB({color[0]}, {color[1]}, {color[2]})', fontsize=8)
            plt.axis('off')

        plt.tight_layout(pad=0.5)
        plt.savefig(f'./static/extract/{filename}')
        plt.close()
            
        return render_template('extract.html', upload=True, filename=filename, colors=colors, k=k)
    return render_template('extract.html', upload=False)

if __name__ == '__main__':
    app.run(debug=True)
