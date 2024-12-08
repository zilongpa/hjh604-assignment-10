from flask import Flask, render_template, request
from PIL import Image
from open_clip import create_model_and_transforms, tokenizer
import torch.nn.functional as F
import pandas as pd
import open_clip
import torch
import os
from flask import send_from_directory
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import euclidean_distances
import numpy as np

df = pd.read_pickle('image_embeddings.pickle')
model, _, preprocess = create_model_and_transforms(
    'ViT-B/32', pretrained='openai')
tokenizer = open_clip.get_tokenizer('ViT-B-32')
model.eval()
app = Flask(__name__)


def load_images(image_dir, max_images=None, target_size=(224, 224)):
    images = []
    image_names = []
    for i, filename in enumerate(os.listdir(image_dir)):
        if filename.endswith('.jpg'):
            img = Image.open(os.path.join(image_dir, filename))
            img = img.convert('L')  # Convert to grayscale ('L' mode)
            img = img.resize(target_size)  # Resize to target size
            # Normalize pixel values to [0, 1]
            img_array = np.asarray(img, dtype=np.float32) / 255.0
            images.append(img_array.flatten())  # Flatten to 1D
            image_names.append(filename)
        if max_images and i + 1 >= max_images:
            break
    return np.array(images), image_names


@app.route('/coco_images_resized/<filename>')
def serve_image(filename):
    return send_from_directory(os.path.join(app.root_path, 'coco_images_resized'), filename)


@app.route('/', methods=['GET', 'POST'])
def main():
    if request.method == 'GET':
        return render_template("index.html")
    elif request.method == 'POST':
        text = request.form.get('text')
        image = request.files['image']
        weight = float(request.form.get('weight'))
        type_ = request.form.get('type')
        k = request.form.get('k')
        if k != '':
            k = int(k)
            if k > 0:
                train_images, train_image_names = load_images(
                    'coco_images_resized', max_images=2000, target_size=(224, 224))
                pca = PCA(n_components=k)
                pca.fit(train_images)
                print('PCA fitted')
                transform_images, transform_image_names = load_images(
                    'coco_images_resized', max_images=10000, target_size=(224, 224))  # This is too slow so only 10000 images are used
                reduced_embeddings = pca.transform(transform_images)
                img = Image.open(image)
                img = img.convert('L')
                img = img.resize((224, 224))
                img_array = np.asarray(img, dtype=np.float32) / 255.0
                query_embedding = pca.transform(img_array.flatten().reshape(1, -1))
                print('PCA transformed')
                distances = euclidean_distances(
                    query_embedding.reshape(1, -1), reduced_embeddings).flatten()
                nearest_indices = np.argsort(distances)[:5]
                results = [{'image': f"coco_images_resized/{transform_image_names[i]}",
                            'similarity': f"Distance: {distances[i]}"} for i in nearest_indices]
                return render_template("index.html", results=results)
        if text:
            text = tokenizer([text])
            text_query = F.normalize(model.encode_text(text))
        if image:
            image = preprocess(Image.open(image)).unsqueeze(0)
            image_query = F.normalize(model.encode_image(image))

        if type_ == 'text':
            similarities = df['embedding'].apply(lambda x: F.cosine_similarity(
                text_query, torch.tensor(x).unsqueeze(0)).item())
        elif type_ == 'image':
            similarities = df['embedding'].apply(lambda x: F.cosine_similarity(
                image_query, torch.tensor(x).unsqueeze(0)).item())
        elif type_ == 'hybrid':
            query_embedding = F.normalize(
                weight * text_query + (1.0 - weight) * image_query)
            similarities = df['embedding'].apply(lambda x: F.cosine_similarity(
                query_embedding, torch.tensor(x).unsqueeze(0)).item())

        top_5_indices = similarities.nlargest(5).index
        top_5_images = df.loc[top_5_indices]['file_name'].tolist()
        results = [{'image': f"coco_images_resized/{top_5_images[i]}",
                   'similarity': f"Similarity: {similarities[top_5_indices[i]]}"} for i in range(5)]
        return render_template("index.html", results=results)
