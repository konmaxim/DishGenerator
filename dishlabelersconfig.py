import os
import requests
from PIL import Image
from io import BytesIO
from tqdm import tqdm
import pandas as pd

# ------------------ CONFIG -------------------

DATA_DIR = "food101_dataset"
IMAGES_DIR = os.path.join(DATA_DIR, "images")
CSV_FILE = os.path.join(DATA_DIR, "labels.csv")

IMAGE_SIZE = (224, 224)
MAX_IMAGES_PER_CLASS = 100  # Adjust as needed or set None for all available

# Full Food-101 class tags (cuisine, dish_type, nutrition, meal_type)
FOOD101_CLASS_TAGS = {
    "apple_pie":           {"cuisine": "american",    "dish_type": "dessert",     "nutrition": "high-cal",   "meal_type": "snack"},
    "baby_back_ribs":      {"cuisine": "american",    "dish_type": "main",        "nutrition": "fatty",      "meal_type": "dinner"},
    "baklava":             {"cuisine": "middle_east", "dish_type": "dessert",     "nutrition": "high-cal",   "meal_type": "snack"},
    "beef_carpaccio":      {"cuisine": "italian",     "dish_type": "appetizer",   "nutrition": "fatty",      "meal_type": "dinner"},
    "beef_tartare":        {"cuisine": "french",      "dish_type": "appetizer",   "nutrition": "fatty",      "meal_type": "dinner"},
    "beef_wellington":     {"cuisine": "british",     "dish_type": "main",        "nutrition": "fatty",      "meal_type": "dinner"},
    "beer":                {"cuisine": "various",     "dish_type": "drink",       "nutrition": "high-cal",   "meal_type": "snack"},
    "bibimbap":            {"cuisine": "korean",      "dish_type": "main",        "nutrition": "healthy",    "meal_type": "lunch"},
    "bread_pudding":       {"cuisine": "british",     "dish_type": "dessert",     "nutrition": "high-cal",   "meal_type": "dessert"},
    "breakfast_burrito":   {"cuisine": "american",    "dish_type": "main",        "nutrition": "high-cal",   "meal_type": "breakfast"},
    "bruschetta":          {"cuisine": "italian",     "dish_type": "appetizer",   "nutrition": "healthy",    "meal_type": "snack"},
    "caesar_salad":        {"cuisine": "american",    "dish_type": "salad",       "nutrition": "healthy",    "meal_type": "lunch"},
    "cannoli":             {"cuisine": "italian",     "dish_type": "dessert",     "nutrition": "high-cal",   "meal_type": "dessert"},
    "caprese_salad":       {"cuisine": "italian",     "dish_type": "salad",       "nutrition": "healthy",    "meal_type": "lunch"},
    "carrot_cake":         {"cuisine": "american",    "dish_type": "dessert",     "nutrition": "high-cal",   "meal_type": "dessert"},
    "ceviche":             {"cuisine": "peruvian",    "dish_type": "appetizer",   "nutrition": "healthy",    "meal_type": "lunch"},
    "cheesecake":          {"cuisine": "american",    "dish_type": "dessert",     "nutrition": "high-cal",   "meal_type": "dessert"},
    "cheese_plate":        {"cuisine": "french",      "dish_type": "appetizer",   "nutrition": "fatty",      "meal_type": "snack"},
    "chicken_curry":       {"cuisine": "indian",      "dish_type": "main",        "nutrition": "fatty",      "meal_type": "dinner"},
    "chicken_quesadilla":  {"cuisine": "mexican",     "dish_type": "main",        "nutrition": "high-cal",   "meal_type": "lunch"},
    "chicken_wings":       {"cuisine": "american",    "dish_type": "appetizer",   "nutrition": "fatty",      "meal_type": "snack"},
    "chocolate_cake":      {"cuisine": "american",    "dish_type": "dessert",     "nutrition": "high-cal",   "meal_type": "dessert"},
    "chocolate_mousse":    {"cuisine": "french",      "dish_type": "dessert",     "nutrition": "high-cal",   "meal_type": "dessert"},
    "churros":             {"cuisine": "spanish",     "dish_type": "dessert",     "nutrition": "high-cal",   "meal_type": "snack"},
    "clam_chowder":        {"cuisine": "american",    "dish_type": "soup",        "nutrition": "fatty",      "meal_type": "lunch"},
    "club_sandwich":       {"cuisine": "american",    "dish_type": "main",        "nutrition": "high-cal",   "meal_type": "lunch"},
    "crab_cakes":          {"cuisine": "american",    "dish_type": "appetizer",   "nutrition": "fatty",      "meal_type": "dinner"},
    "creme_brulee":        {"cuisine": "french",      "dish_type": "dessert",     "nutrition": "high-cal",   "meal_type": "dessert"},
    "croque_madame":       {"cuisine": "french",      "dish_type": "main",        "nutrition": "high-cal",   "meal_type": "lunch"},
    "cup_cakes":           {"cuisine": "american",    "dish_type": "dessert",     "nutrition": "high-cal",   "meal_type": "snack"},
    "deviled_eggs":        {"cuisine": "american",    "dish_type": "appetizer",   "nutrition": "fatty",      "meal_type": "snack"},
    "donuts":              {"cuisine": "american",    "dish_type": "dessert",     "nutrition": "high-cal",   "meal_type": "snack"},
    "dumplings":           {"cuisine": "chinese",     "dish_type": "main",        "nutrition": "fatty",      "meal_type": "dinner"},
    "edamame":             {"cuisine": "japanese",    "dish_type": "appetizer",   "nutrition": "healthy",    "meal_type": "snack"},
    "eggs_benedict":       {"cuisine": "american",    "dish_type": "breakfast",   "nutrition": "high-cal",   "meal_type": "breakfast"},
    "escargots":           {"cuisine": "french",      "dish_type": "appetizer",   "nutrition": "fatty",      "meal_type": "snack"},
    "falafel":             {"cuisine": "middle_east", "dish_type": "appetizer",   "nutrition": "healthy",    "meal_type": "snack"},
    "filet_mignon":        {"cuisine": "french",      "dish_type": "main",        "nutrition": "fatty",      "meal_type": "dinner"},
    "fish_and_chips":      {"cuisine": "british",     "dish_type": "main",        "nutrition": "fatty",      "meal_type": "dinner"},
    "foie_gras":           {"cuisine": "french",      "dish_type": "appetizer",   "nutrition": "fatty",      "meal_type": "snack"},
    "french_fries":        {"cuisine": "american",    "dish_type": "side",        "nutrition": "high-cal",   "meal_type": "snack"},
    "french_onion_soup":   {"cuisine": "french",      "dish_type": "soup",        "nutrition": "fatty",      "meal_type": "lunch"},
    "french_toast":        {"cuisine": "american",    "dish_type": "breakfast",   "nutrition": "high-cal",   "meal_type": "breakfast"},
    "fried_calamari":      {"cuisine": "italian",     "dish_type": "appetizer",   "nutrition": "fatty",      "meal_type": "snack"},
    "fried_rice":          {"cuisine": "chinese",     "dish_type": "main",        "nutrition": "high-cal",   "meal_type": "lunch"},
    "fried_shrimp":        {"cuisine": "american",    "dish_type": "appetizer",   "nutrition": "fatty",      "meal_type": "snack"},
    "guacamole":           {"cuisine": "mexican",     "dish_type": "appetizer",   "nutrition": "healthy",    "meal_type": "snack"},
    "gyoza":               {"cuisine": "japanese",    "dish_type": "appetizer",   "nutrition": "fatty",      "meal_type": "snack"},
    "hamburger":           {"cuisine": "american",    "dish_type": "main",        "nutrition": "high-cal",   "meal_type": "dinner"},
    "hot_and_sour_soup":   {"cuisine": "chinese",     "dish_type": "soup",        "nutrition": "healthy",    "meal_type": "lunch"},
    "hot_dog":             {"cuisine": "american",    "dish_type": "main",        "nutrition": "high-cal",   "meal_type": "lunch"},
    "huevos_rancheros":    {"cuisine": "mexican",     "dish_type": "breakfast",   "nutrition": "high-cal",   "meal_type": "breakfast"},
    "hummus":              {"cuisine": "middle_east", "dish_type": "appetizer",   "nutrition": "healthy",    "meal_type": "snack"},
    "ice_cream":           {"cuisine": "various",     "dish_type": "dessert",     "nutrition": "high-cal",   "meal_type": "dessert"},
    "lasagna":             {"cuisine": "italian",     "dish_type": "main",        "nutrition": "high-cal",   "meal_type": "dinner"},
    "lobster_bisque":      {"cuisine": "french",      "dish_type": "soup",        "nutrition": "fatty",      "meal_type": "lunch"},
    "lobster_roll_sandwich":{"cuisine":"american",    "dish_type": "main",        "nutrition": "fatty",      "meal_type": "lunch"},
    "macaroni_and_cheese": {"cuisine": "american",    "dish_type": "main",        "nutrition": "high-cal",   "meal_type": "dinner"},
    "macarons":            {"cuisine": "french",      "dish_type": "dessert",     "nutrition": "high-cal",   "meal_type": "snack"},
    "miso_soup":           {"cuisine": "japanese",    "dish_type": "soup",        "nutrition": "healthy",    "meal_type": "lunch"},
    "mussels":             {"cuisine": "french",      "dish_type": "appetizer",   "nutrition": "healthy",    "meal_type": "dinner"},
    "nachos":              {"cuisine": "mexican",     "dish_type": "appetizer",   "nutrition": "high-cal",   "meal_type": "snack"},
    "omelette":            {"cuisine": "french",      "dish_type": "breakfast",   "nutrition": "healthy",    "meal_type": "breakfast"},
    "onion_rings":         {"cuisine": "american",    "dish_type": "side",        "nutrition": "high-cal",   "meal_type": "snack"},
    "oysters":             {"cuisine": "french",      "dish_type": "appetizer",   "nutrition": "healthy",    "meal_type": "dinner"},
    "pad_thai":            {"cuisine": "thai",        "dish_type": "main",        "nutrition": "healthy",    "meal_type": "dinner"},
    "paella":              {"cuisine": "spanish",     "dish_type": "main",        "nutrition": "fatty",      "meal_type": "dinner"},
    "pancakes":            {"cuisine": "american",    "dish_type": "breakfast",   "nutrition": "high-cal",   "meal_type": "breakfast"},
    "panna_cotta":         {"cuisine": "italian",     "dish_type": "dessert",     "nutrition": "high-cal",   "meal_type": "dessert"},
    "peking_duck":         {"cuisine": "chinese",     "dish_type": "main",        "nutrition": "fatty",      "meal_type": "dinner"},
    "pho":                 {"cuisine": "vietnamese",  "dish_type": "soup",        "nutrition": "healthy",    "meal_type": "lunch"},
    "pizza":               {"cuisine": "italian",     "dish_type": "main",        "nutrition": "high-cal",   "meal_type": "dinner"},
    "pork_chop":           {"cuisine": "american",    "dish_type": "main",        "nutrition": "fatty",      "meal_type": "dinner"},
    "poutine":             {"cuisine": "canadian",    "dish_type": "side",        "nutrition": "high-cal",   "meal_type": "snack"},
    "prime_rib":           {"cuisine": "american",    "dish_type": "main",        "nutrition": "fatty",      "meal_type": "dinner"},
    "pulled_pork_sandwich":{"cuisine": "american",    "dish_type": "main",        "nutrition": "fatty",      "meal_type": "lunch"},
    "ramen":               {"cuisine": "japanese",    "dish_type": "soup",        "nutrition": "healthy",    "meal_type": "dinner"},
    "risotto":             {"cuisine": "italian",     "dish_type": "main",        "nutrition": "high-cal",   "meal_type": "dinner"},
    "samosa":              {"cuisine": "indian",      "dish_type": "appetizer",   "nutrition": "high-cal",   "meal_type": "snack"},
    "sashimi":             {"cuisine": "japanese",    "dish_type": "main",        "nutrition": "healthy",    "meal_type": "dinner"},
    "scallops":            {"cuisine": "french",      "dish_type": "appetizer",   "nutrition": "healthy",    "meal_type": "dinner"},
    "seaweed_salad":       {"cuisine": "japanese",    "dish_type": "salad",       "nutrition": "healthy",    "meal_type": "lunch"},
    "shrimp_and_grits":    {"cuisine": "southern_us","dish_type": "main",        "nutrition": "fatty",      "meal_type": "dinner"},
    "spaghetti_bolognese": {"cuisine": "italian",     "dish_type": "main",        "nutrition": "high-cal",   "meal_type": "dinner"},
    "spaghetti_carbonara": {"cuisine": "italian",     "dish_type": "main",        "nutrition": "high-cal",   "meal_type": "dinner"},
    "spring_rolls":        {"cuisine": "chinese",     "dish_type": "appetizer",   "nutrition": "healthy",    "meal_type": "snack"},
    "steak":               {"cuisine": "american",    "dish_type": "main",        "nutrition": "fatty",      "meal_type": "dinner"},
    "strawberry_shortcake":{"cuisine": "american",    "dish_type": "dessert",     "nutrition": "high-cal",   "meal_type": "dessert"},
    "sushi":               {"cuisine": "japanese",    "dish_type": "main",        "nutrition": "healthy",    "meal_type": "lunch"},
    "tacos":               {"cuisine": "mexican",     "dish_type": "main",        "nutrition": "high-cal",   "meal_type": "dinner"},
    "takoyaki":            {"cuisine": "japanese",    "dish_type": "snack",       "nutrition": "high-cal",   "meal_type": "snack"},
    "tiramisu":            {"cuisine": "italian",     "dish_type": "dessert",     "nutrition": "high-cal",   "meal_type": "dessert"},
    "tuna_tartare":        {"cuisine": "american",    "dish_type": "appetizer",   "nutrition": "healthy",    "meal_type": "dinner"},
    "waffles":             {"cuisine": "belgian",     "dish_type": "breakfast",   "nutrition": "high-cal",   "meal_type": "breakfast"},
}

# URL to Food-101 images JSON file (hosted on GitHub)
FOOD101_URL = "https://raw.githubusercontent.com/food101-dataset/food101/master/meta/train.txt"

# Helper to create directories if missing
def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

# Resize and save image
def save_resized_image(image_url, save_path):
    try:
        response = requests.get(image_url, timeout=10)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content)).convert("RGB")
        img = img.resize(IMAGE_SIZE)
        img.save(save_path)
        return True
    except Exception as e:
        print(f"Failed to download or process image {image_url}: {e}")
        return False

def main():
    ensure_dir(IMAGES_DIR)
    
    
    base_image_url = "https://data.vision.ee.ethz.ch/cvl/food-101/images"
    
    # Load train.txt with class/image_name lines
    print("Downloading train.txt metadata")
    train_txt = requests.get(FOOD101_URL).text.strip().split('\n')
    
    # image records  for CSV
    records = []
    
    # Count downloaded images per class to limit
    class_counts = {}
    
    for line in tqdm(train_txt):
        if not line.strip() or '/' not in line:
            continue  # skip empty or invalid lines
        class_name, image_name = line.split('/')
        
        if MAX_IMAGES_PER_CLASS:
            if class_counts.get(class_name, 0) >= MAX_IMAGES_PER_CLASS:
                continue
        
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        image_url = f"{base_image_url}/{class_name}/{image_name}.jpg"
        class_dir = os.path.join(IMAGES_DIR, class_name)
        ensure_dir(class_dir)
        save_path = os.path.join(class_dir, f"{image_name}.jpg")
        
        success = save_resized_image(image_url, save_path)
        if not success:
            continue
        
        tags = FOOD101_CLASS_TAGS.get(class_name, {})
        
        records.append({
            "filename": os.path.relpath(save_path, DATA_DIR),
            "class": class_name,
            "cuisine": tags.get("cuisine", ""),
            "dish_type": tags.get("dish_type", ""),
            "nutrition": tags.get("nutrition", ""),
            "meal_type": tags.get("meal_type", ""),
        })
    
    # Save CSV file with labels and tags
    df = pd.DataFrame(records)
    df.to_csv(CSV_FILE, index=False)
    print(f"Done! Dataset and labels saved in {DATA_DIR}")

if __name__ == "__main__":
    main()
