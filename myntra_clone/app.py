from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import os
from werkzeug.utils import secure_filename
from viton_integration import load_model, perform_try_on  # Ensure viton_integration is correctly defined
import argparse
import os
import flash
import torch
from torch import nn
from torch.nn import functional as F
import torchgeometry as tgm

from datasets import VITONDataset, VITONDataLoader
from networks import SegGenerator, GMM, ALIASGenerator
from utils import gen_noise, load_checkpoint, save_images

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
output_folder='static/output'
model_path = "C:/Users/aditi/Documents/VITON-HD-20240712T134614Z-001/checkpoints"
seg, gmm, alias = load_model(model_path)

# Load the dataset from a CSV file
def load_products():
    df = pd.read_csv('myntra_sheet.csv')
    products = df.to_dict(orient='records')
    return products

@app.route('/')
def home():
    products = load_products()
    return render_template('index.html', products=products)

@app.route('/product/<int:product_id>')
def product_page(product_id):
    products = load_products()
    product = next((p for p in products if p['id'] == product_id), None)
    if product:
        return render_template('product.html', product=product)
    else:
        return "Product not found", 404

@app.route('/tryon/<int:product_id>', methods=['GET', 'POST'])
def vton(product_id):
    products = load_products()
    product = next((item for item in products if item['id'] == product_id), None)
    
    # if request.method == 'POST':
    #     if 'userImage' not in request.files:
    #         flash('No file part')
    #         return redirect(request.url)
    #     file = request.files['userImage']
    #     if file.filename == '':
    #         flash('No selected file')
    #         return redirect(request.url)
    #     if file:
    #         filename = secure_filename(file.filename)
    #         print(filename)
    #         user_image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    #         print(user_image_path)
    #         file.save(user_image_path)
    #         print(f"User image saved to {user_image_path}")
            
    #         return redirect(url_for('vton_result', product_id=product_id, user_image=user_image_path))

    return render_template('vton.html', product=product)


@app.route('/vton_result/<int:product_id>', methods=['GET', 'POST'])
def vton_result(product_id):
    if request.method == 'POST':
        if 'userImage' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['userImage']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file:
            filename = secure_filename(file.filename)
            user_image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(user_image_path)
            print(f"User image saved to {user_image_path}")
            
            products = load_products()
            product = next((p for p in products if p['id'] == product_id), None)
            
            if product:
                path='C:/Users/aditi/OneDrive/Desktop/myntra/prototype/Myntra_hack/myntra_clone'
                clothes_image_path = os.path.join('static', 'images', os.path.basename(product['img']))
                output_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'output_' + os.path.basename(user_image_path))
                
                perform_try_on(seg, gmm, alias, user_image_path, clothes_image_path, output_image_path)
                
                return render_template('display_result'.html, user_image=user_image_path, output_image=output_image_path, product=product)
            
            return "Product not found", 404

    # If GET request or file handling fails, return an empty form or handle differently
    return render_template('vton_result.html', product_id=product_id)


if __name__ == '__main__':
    app.run(debug=True)
