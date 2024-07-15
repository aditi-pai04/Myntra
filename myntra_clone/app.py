from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import os
from textblob import TextBlob
from werkzeug.utils import secure_filename
import argparse
import os
import flash
import torch
import base64
from torch import nn
from torch.nn import functional as F
import torchgeometry as tgm
from PIL import Image
from datasets import VITONDataset
import matplotlib.pyplot as plt
import io
import json 
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'C:\\Users\\aditi\\OneDrive\\Desktop\\myntra\\prototype\\Myntra_hack\\myntra_clone\\datasets\\test\\openpose-img'
app.config['UPLOADF']='C:/Users/aditi/OneDrive/Desktop/myntra/prototype/Myntra_hack/myntra_clone/datasets/test/openpose-json'
output_folder='static/output'
model_path = "C:/Users/aditi/Documents/VITON-HD-20240712T134614Z-001/checkpoints"

# Load products and reviews data
file_path_products = 'final\\Myntra_hack\\myntra_clone\\myntra_sheet.csv'
file_path_reviews = 'final\\Myntra_hack\\myntra_clone\\myntra_sheet_with_reviews.csv'

import argparse
import os

import torch
from torch import nn
from torch.nn import functional as F
import torchgeometry as tgm

from datasets import VITONDataset, VITONDataLoader
from networks import SegGenerator, GMM, ALIASGenerator
from utils import gen_noise, load_checkpoint, save_images

# params = {
#     'model_folder': 'C:/Users/aditi/OneDrive/Desktop/myntra/prototype/Myntra_hack/myntra_clone/openpose/models',  # Replace with your OpenPose model folder
#     'disable_blending': False,
#     'number_people_max': 1,
#     'model_pose': 'BODY_25',  # Example, use 'BODY_25' or other models based on your needs
#     'display': 0
# }

# # # Initialize OpenPose
# openpose = op.WrapperPython()
# openpose.configure(params)
# openpose.start()

# def generate_keypoints_openpose(image_path):
#     datum = op.Datum()
#     image_to_process = op.imread(image_path)
#     datum.cvInputData = image_to_process
#     openpose.emplaceAndPop([datum])
    
#     keypoints_data = {
#         "version": 1.3,
#         "people": []
#     }
    
#     for person in datum.poseKeypoints:
#         keypoints_data["people"].append({
#             "person_id": [-1],
#             "pose_keypoints_2d": person.tolist(),
#             "face_keypoints_2d": [],
#             "hand_left_keypoints_2d": [],
#             "hand_right_keypoints_2d": [],
#             "pose_keypoints_3d": [],
#             "face_keypoints_3d": [],
#             "hand_left_keypoints_3d": [],
#             "hand_right_keypoints_3d": []
#         })
#     # keypoints_filename = os.path.splitext(os.path.basename(user_image_path))[0] + '_keypoints.json'
#     # keypoints_save_path = os.path.join(os.path.dirname(save_path), keypoints_filename)
    
#     # with open(keypoints_save_path, 'w') as f:
#     #     json.dump(keypoints_data, f)
#     return keypoints_data

# Classify sentiment
def classify_sentiment(review):
    analysis = TextBlob(str(review))
    if analysis.sentiment.polarity > 0:
        return 'positive'
    elif analysis.sentiment.polarity == 0:
        return 'neutral'
    else:
        return 'negative'

# Summarize reviews
def summarize_reviews(reviews):
    string_reviews = [str(review) for review in reviews]
    summary = TextBlob(" ".join(string_reviews)).noun_phrases
    return ". ".join(summary[:5])  # Take the first 5 key phrases for summary
def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, required=True)

    parser.add_argument('-b', '--batch_size', type=int, default=1)
    parser.add_argument('-j', '--workers', type=int, default=1)
    parser.add_argument('--load_height', type=int, default=1024)
    parser.add_argument('--load_width', type=int, default=768)
    parser.add_argument('--shuffle', action='store_true')

    parser.add_argument('--dataset_dir', type=str, default='./datasets/')
    parser.add_argument('--dataset_mode', type=str, default='test')
    parser.add_argument('--dataset_list', type=str, default='test_pairs.txt')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints/')
    parser.add_argument('--save_dir', type=str, default='./results/')

    parser.add_argument('--display_freq', type=int, default=1)

    parser.add_argument('--seg_checkpoint', type=str, default='seg_final.pth')
    parser.add_argument('--gmm_checkpoint', type=str, default='gmm_final.pth')
    parser.add_argument('--alias_checkpoint', type=str, default='alias_final.pth')

    # common
    parser.add_argument('--semantic_nc', type=int, default=13, help='# of human-parsing map classes')
    parser.add_argument('--init_type', choices=['normal', 'xavier', 'xavier_uniform', 'kaiming', 'orthogonal', 'none'], default='xavier')
    parser.add_argument('--init_variance', type=float, default=0.02, help='variance of the initialization distribution')

    # for GMM
    parser.add_argument('--grid_size', type=int, default=5)

    # for ALIASGenerator
    parser.add_argument('--norm_G', type=str, default='spectralaliasinstance')
    parser.add_argument('--ngf', type=int, default=64, help='# of generator filters in the first conv layer')
    parser.add_argument('--num_upsampling_layers', choices=['normal', 'more', 'most'], default='most',
                        help='If \'more\', add upsampling layer between the two middle resnet blocks. '
                             'If \'most\', also add one more (upsampling + resnet) layer at the end of the generator.')

    opt = parser.parse_args()
    return opt

opt = get_opt()
print(opt)

if not os.path.exists(os.path.join(opt.save_dir, opt.name)):
    os.makedirs(os.path.join(opt.save_dir, opt.name))

seg = SegGenerator(opt, input_nc=opt.semantic_nc + 8, output_nc=opt.semantic_nc)
gmm = GMM(opt, inputA_nc=7, inputB_nc=3)
opt.semantic_nc = 7
alias = ALIASGenerator(opt, input_nc=9)
opt.semantic_nc = 13

load_checkpoint(seg, 'C:/Users/aditi/OneDrive/Desktop/myntra/prototype/Myntra_hack/VITON-HD/checkpoints/seg_final.pth')
load_checkpoint(gmm, 'C:/Users/aditi/OneDrive/Desktop/myntra/prototype/Myntra_hack/VITON-HD/checkpoints/gmm_final.pth')
load_checkpoint(alias, 'C:/Users/aditi/OneDrive/Desktop/myntra/prototype/Myntra_hack/VITON-HD/checkpoints/alias_final.pth')

def test(opt, seg, gmm, alias):
    up = nn.Upsample(size=(opt.load_height, opt.load_width), mode='bilinear')
    gauss = tgm.image.GaussianBlur((15, 15), (3, 3))
    gauss

    test_dataset = VITONDataset(opt)
    test_loader = VITONDataLoader(opt, test_dataset)

    with torch.no_grad():
        for i, inputs in enumerate(test_loader.data_loader):
            img_names = inputs['img_name']
            c_names = inputs['c_name']['unpaired']

            img_agnostic = inputs['img_agnostic']
            parse_agnostic = inputs['parse_agnostic']
            pose = inputs['pose']
            c = inputs['cloth']['unpaired']
            cm = inputs['cloth_mask']['unpaired']

            # Part 1. Segmentation generation
            parse_agnostic_down = F.interpolate(parse_agnostic, size=(256, 192), mode='bilinear')
            pose_down = F.interpolate(pose, size=(256, 192), mode='bilinear')
            c_masked_down = F.interpolate(c * cm, size=(256, 192), mode='bilinear')
            cm_down = F.interpolate(cm, size=(256, 192), mode='bilinear')
            seg_input = torch.cat((cm_down, c_masked_down, parse_agnostic_down, pose_down, gen_noise(cm_down.size())), dim=1)

            parse_pred_down = seg(seg_input)
            parse_pred = gauss(up(parse_pred_down))
            parse_pred = parse_pred.argmax(dim=1)[:, None]

            parse_old = torch.zeros(parse_pred.size(0), 13, opt.load_height, opt.load_width, dtype=torch.float)
            parse_old.scatter_(1, parse_pred, 1.0)

            labels = {
                0:  ['background',  [0]],
                1:  ['paste',       [2, 4, 7, 8, 9, 10, 11]],
                2:  ['upper',       [3]],
                3:  ['hair',        [1]],
                4:  ['left_arm',    [5]],
                5:  ['right_arm',   [6]],
                6:  ['noise',       [12]]
            }
            parse = torch.zeros(parse_pred.size(0), 7, opt.load_height, opt.load_width, dtype=torch.float)
            for j in range(len(labels)):
                for label in labels[j][1]:
                    parse[:, j] += parse_old[:, label]

            # Part 2. Clothes Deformation
            agnostic_gmm = F.interpolate(img_agnostic, size=(256, 192), mode='nearest')
            parse_cloth_gmm = F.interpolate(parse[:, 2:3], size=(256, 192), mode='nearest')
            pose_gmm = F.interpolate(pose, size=(256, 192), mode='nearest')
            c_gmm = F.interpolate(c, size=(256, 192), mode='nearest')
            gmm_input = torch.cat((parse_cloth_gmm, pose_gmm, agnostic_gmm), dim=1)

            _, warped_grid = gmm(gmm_input, c_gmm)
            warped_c = F.grid_sample(c, warped_grid, padding_mode='border')
            warped_cm = F.grid_sample(cm, warped_grid, padding_mode='border')

            # Part 3. Try-on synthesis
            misalign_mask = parse[:, 2:3] - warped_cm
            misalign_mask[misalign_mask < 0.0] = 0.0
            parse_div = torch.cat((parse, misalign_mask), dim=1)
            parse_div[:, 2:3] -= misalign_mask

            output = alias(torch.cat((img_agnostic, pose, warped_c), dim=1), parse, parse_div, misalign_mask)

            unpaired_names = []
            for img_name, c_name in zip(img_names, c_names):
                unpaired_names.append('{}_{}'.format(img_name.split('_')[0], c_name))

            save_images(output, unpaired_names, os.path.join(opt.save_dir, opt.name))

            if (i + 1) % opt.display_freq == 0:
                print("step: {}".format(i + 1))


def main():
    opt = get_opt()
    print(opt)

    if not os.path.exists(os.path.join(opt.save_dir, opt.name)):
        os.makedirs(os.path.join(opt.save_dir, opt.name))

    seg = SegGenerator(opt, input_nc=opt.semantic_nc + 8, output_nc=opt.semantic_nc)
    gmm = GMM(opt, inputA_nc=7, inputB_nc=3)
    opt.semantic_nc = 7
    alias = ALIASGenerator(opt, input_nc=9)
    opt.semantic_nc = 13

    load_checkpoint(seg, 'C:/Users/aditi/OneDrive/Desktop/myntra/prototype/Myntra_hack/VITON-HD/checkpoints/seg_final.pth')
    load_checkpoint(gmm, 'C:/Users/aditi/OneDrive/Desktop/myntra/prototype/Myntra_hack/VITON-HD/checkpoints/gmm_final.pth')
    load_checkpoint(alias, 'C:/Users/aditi/OneDrive/Desktop/myntra/prototype/Myntra_hack/VITON-HD/checkpoints/alias_final.pth')

    seg.eval()
    gmm.eval()
    alias.eval()
    test(opt, seg, gmm, alias)

# Load the dataset from a CSV file
def load_products():
    df = pd.read_csv('myntra_sheet.csv')
    products = df.to_dict(orient='records')
    return products

@app.route('/')
def home():
    products = load_products()
    return render_template('index.html', products=products)

# @app.route('/product/<int:product_id>')
# def product_page(product_id):
#     products = load_products()
#     product = next((p for p in products if p['id'] == product_id), None)
#     if product:
#         return render_template('product.html', product=product)
#     else:
#         return "Product not found", 404

reviews_df = pd.read_csv('C:\\Users\\aditi\\OneDrive\\Desktop\\myntra\\prototype\\Myntra_hack\\myntra_clone\\myntra_sheet_with_reviews.csv')

@app.route('/product/<int:product_id>')
def product_page(product_id):
    products = load_products()
    product = next((p for p in products if p['id'] == product_id), None)
    
    if not product:
        return "Product not found", 404

    # Filter reviews for the specific product ID
    product_reviews = reviews_df[reviews_df['id'] == product_id]['comment']

    return render_template(
        'product.html',
        product=product,
        image=product['img'],
        product_reviews=product_reviews,
    )

@app.route('/summary/<int:product_id>')
def review_summary(product_id):
    # Filter reviews for the specific product ID
    product_reviews = reviews_df[reviews_df['id'] == product_id]
    
    # Initialize sentiment counts
    positive_count = 0
    neutral_count = 0
    negative_count = 0

    positive_reviews = []
    neutral_reviews = []
    negative_reviews = []

    for review in product_reviews['comment']:
        sentiment = classify_sentiment(review)
        if sentiment == 'positive':
            positive_reviews.append(review)
            positive_count += 1
        elif sentiment == 'neutral':
            neutral_reviews.append(review)
            neutral_count += 1
        else:
            negative_reviews.append(review)
            negative_count += 1

    # Summarize reviews for each sentiment
    summary_positive = summarize_reviews(positive_reviews)
    summary_neutral = summarize_reviews(neutral_reviews)
    summary_negative = summarize_reviews(negative_reviews)

    # Create pie chart
    counts = [positive_count, neutral_count, negative_count]
    labels = ['Positive', 'Neutral', 'Negative']
    colors = ['green', 'orange', 'red']
    
    fig, ax = plt.subplots()
    ax.pie(counts, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
    ax.axis('equal')
    plt.title(f'Sentiment Distribution for Product ID {product_id}')

    # Save plot to a string buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    chart_data = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()

    return render_template(
        'summary.html',
        product_id=product_id,
        summary_positive=summary_positive,
        summary_neutral=summary_neutral,
        summary_negative=summary_negative,
        chart_data=chart_data,
    )

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
    #         cloth_image_path=product['img']
    #         with open(r'C:\\Users\\aditi\\OneDrive\\Desktop\\myntra\\prototype\\Myntra_hack\\VITON-HD\\datasets\\test_pairs.txt', 'w') as f:
    #             f.write(f"{os.path.basename(user_image_path)} {os.path.basename(cloth_image_path)}\n")
    #             print("Fiel changes")
    #         print(f"User image saved to {user_image_path}")
            
    #         return redirect(url_for('vton_result', product_id=product_id, user_image=user_image_path))

    return render_template('vton.html', product=product)

@app.route('/vton_result/<int:product_id>', methods=['POST'])
def vton_result(product_id):
    products = load_products()
    product = next((item for item in products if item['id'] == product_id), None)
    if request.method == 'POST':
        if 'userImage' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['userImage']
        keypoints = request.form['keypoints']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file:
            filename = secure_filename(file.filename)
            print(filename)
            user_image_path = os.path.join(app.config['UPLOAD_FOLDER'], os.path.splitext(filename)[0] + '.png')
            print(user_image_path)
            file.save(user_image_path)
            # keypoints_filename = os.path.splitext(os.path.basename(filename))[0] + '_keypoints.json'
            # keypoints_data = generate_keypoints_openpose(user_image_path)
            # save_path='C:\\Users\\aditi\\OneDrive\\Desktop\\myntra\\prototype\\Myntra_hack\\myntra_clone\\datasets\\test\\openpose-json'
            # keypoints_save_path = os.path.join(save_path, keypoints_filename)
            # with open(keypoints_save_path, 'w') as f:
            #     json.dump(keypoints_data, f)
            keypoints_path = os.path.join(app.config['UPLOADF'], f"{os.path.splitext(filename)[0]}_keypoints.json")
            with open(keypoints_path, 'w') as f:
                f.write(keypoints)
            cloth_image_path=product['img']
            with open(r'C:/Users/aditi/OneDrive/Desktop/myntra/prototype/Myntra_hack/myntra_clone/datasets/test_pairs.txt', 'w') as f:
                f.write(f"{os.path.basename(user_image_path)} {os.path.basename(cloth_image_path)}\n")
                print("Fiel changes")
            print(f"User image saved to {user_image_path}")
    # print(product_id)
    # #image=request.args.get('userImage')
    # products = load_products()
    # product = next((item for item in products if item['id'] == product_id), None)
    test(opt,seg,gmm,alias)

if __name__ == '__main__':
    app.run(debug=True)
