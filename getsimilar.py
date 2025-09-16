from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
import pandas as pd
import numpy as np
import joblib
import faiss
#load embedding

app = Flask(__name__)

df = pd.read_csv("df.csv")
embeddings = np.load("embeddings.npy")  
kmeans = joblib.load("kmeans_model.pkl")
model = SentenceTransformer("all-MiniLM-L6-v2")

df.rename(columns={"RecipeCategory": "Category", "RecipeIngredientParts": "Ingredients", "Images": "Image"}, inplace=True)
df["Ingredients"] = df["Ingredients"].str.replace('"', '', regex=False).str.strip()
cols =  ["Name", "Description", "Image", "Ingredients", "Category"]
df["Image"] = df["Image"].str.extract(r'(https?://\S+\.(?:jpg|png|jpeg|gif))')
#keep only recipes that have images (for now)
img_mask = df["Image"].notna()
df = df[img_mask].reset_index(drop=True)
embeddings = embeddings[img_mask.values]
labels = kmeans.labels_[img_mask] 
#dictionary out of rows
info_to_send = df[cols].to_dict(orient="records")
print(len(info_to_send))
print(embeddings.shape)

@app.route("/Create/embeddishes",methods=["POST"])

#two clusters (based on some EDA)
def embed_dishes():
        dish_texts = request.get_json()
        
        coin = np.random.random()
        if coin < 0.7:
               coin = 0
        else:
               coin = 1 
        mask = (labels == coin)
        random_cluster = embeddings[mask]
        cluster_indices = np.where(mask)[0]
        index = faiss.IndexFlatL2(384)   
        print(index.is_trained)
        index.add(random_cluster)  # add vectors to the index
        print(index.ntotal)
        #optimize later to recieve embeddings at creation 
        user_embeddings = model.encode(dish_texts)
        #load kmeans model 
        user_embeddings = normalize(user_embeddings)
        user_clusters = kmeans.predict(user_embeddings)
        centroids = [np.zeros(384),np.zeros(384)]
        for cl in [0,1]:
        # select embeddings in this cluster
                cl_embs = user_embeddings[user_clusters == cl]
                if len(cl_embs) > 0:
                      centroids[cl]=np.mean(cl_embs, axis=0) 
        #nn search with faiss 
        query = centroids[coin].reshape(1, -1)
        D, I = index.search(query, 5)    
        print(I)                   # indicies of 5-nn
        dish_idx = I[0][np.random.choice([0,1,2,3,4])] #choose random dish
        real_idx = cluster_indices[dish_idx]
        print(info_to_send[real_idx])
        return jsonify(info_to_send[real_idx])


                          
        

