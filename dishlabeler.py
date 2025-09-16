import numpy as np 
import pandas as pd 
import umap
import matplotlib.pyplot  as plt
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
import umap.plot
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import joblib
#Prepare df for analysis  
df = pd.read_csv('recipes.csv', nrows =50000)
df.drop_duplicates(subset=['Name'])
def clean_rows(row):
    df[row] =   (df[row]
    .str.replace(r"^c\(|\)$", "", regex=True)
    .str.replace("'", "", regex=False)
    .str.strip()
    .str.lower())

clean_rows('Keywords')
clean_rows('RecipeIngredientParts')
clean_rows('Images')

df['full_text'] = (
    "Name: " + df['Name'] + " " +
    "Category: " + df['RecipeCategory'] + " " +
    "Keywords: " + df['Keywords'] + " " +
    "Ingredients: " + df['RecipeIngredientParts']
) 
df = df.dropna(subset=["full_text"])
df.to_csv("df.csv")
#-----------------embedding was done on Colab -----------------
#model = SentenceTransformer('all-MiniLM-L6-v2')
#df.drop_duplicates(subset=['Name'])
#texts = df['full_text'].tolist()
#bad_rows = df[~df['full_text'].apply(lambda x: isinstance(x, str))]

#df = df.dropna(subset=['full_text'])
#embeddings = model.encode(df["full_text"].tolist(), convert_to_numpy=True, show_progress_bar = True)
#np.save("recipe_embeddings_small.npy", embeddings)

embeddings = np.load("embeddings.npy") 
print(embeddings.shape)
embeddings_norm = normalize(embeddings)
#We'd like to map the results using UMAP
#try out 50 to find more global similarities 
map = umap.UMAP(n_neighbors = 50, min_dist = 0.01, n_components= 3, metric="cosine")
emb_norm = normalize(embeddings, axis=1)
X_umap = map.fit_transform(emb_norm)

plt.figure(figsize=(8,6))
plt.scatter(X_umap[:,0], X_umap[:,1], s=5, alpha=0.7)
plt.title("UMAP projection of embeddings")
plt.savefig("outputs/umap_projection.png", dpi=300, bbox_inches='tight')
plt.show()
#we'll choose a k=2, based on UMAP projection showing two blobs with seperation 
#further analysis showed no structure of subclasses inside cluster 0 (the savory cluster)
k = 2  # number of clusters
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
kmeans.fit(embeddings_norm)
centers = kmeans.cluster_centers_
labels = kmeans.labels_
#save for later
joblib.dump(kmeans, "kmeans_model.pkl")
# luster centers and labels



#just to test  
df_clusters = pd.DataFrame({
    "Name": df["Name"].values,
    "Cluster": labels
})

df_clusters.to_csv("dish_clusters.csv", index=False)

#reduce dimensionality to check for possible subcluster structure 
labels = kmeans.fit_predict(embeddings)

# attach cluster labels back to df
df["cluster"] = labels
#masks
cluster0_mask = labels == 0
cluster0_embs = embeddings[cluster0_mask]
#check possible subcluster structure
reducer = umap.UMAP(n_neighbors=20, min_dist=0.1, metric='cosine', random_state=42)
cluster0_2d = reducer.fit_transform(cluster0_embs)

plt.figure(figsize=(8,6))
scatter = plt.scatter(cluster0_2d[:,0], cluster0_2d[:,1], cmap='tab20', s=10)
plt.title("Subclusters inside Cluster 0 (UMAP 2D)")
plt.xlabel("UMAP1")
plt.ylabel("UMAP2")
plt.savefig("outputs/cluster_0_projection.png", dpi=300, bbox_inches='tight')
plt.show()