import os
import sys
import time
import hashlib
import requests
from io import BytesIO
from PIL import Image

CLASSES = {
    "cup": [
        "coffee mug",
        "coffee cup desk",
        "tea mug table",
        "ceramic mug table",
        "drinking mug desk",
        "ceramic cup",
        "mug table office",
        "cup saucer table"
    ],
    "keys": [
        "keychain keys",
        "house keys table",
        "car keys desk",
        "bunch of keys",
        "set of keys table",
        "door keys keychain",
        "keys ring desk",
        "padlock keys"
    ],
    "headphones": [
        "headphones desk",
        "over ear headphones",
        "audio headphones",
        "wireless headphones desk",
        "studio headphones table",
        "earphones desk",
        "black headphones table",
        "headset table"
    ],
    "pen": [
        "ballpoint pen",
        "ballpoint pen desk",
        "fountain pen table",
        "writing pen notebook",
        "pen on desk",
        "gel pen table",
        "pen paper desk",
        "pencils and pens desk"
    ]
}

TARGET_PER_CLASS = 120
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "dataset", "raw")

def fetch_wikimedia_images(query, max_results=50):
    url = "https://commons.wikimedia.org/w/api.php"
    params = {
        "action": "query",
        "generator": "search",
        "gsrsearch": query,
        "gsrlimit": max_results,
        "gsrnamespace": 6, # File namespace
        "prop": "imageinfo",
        "iiprop": "url",
        "iiurlwidth": 640, # Get optimized 640px web thumbnail
        "format": "json"
    }
    headers = {
        "User-Agent": "UniversityDesktopCVDetectionBot/1.0 (academic; student research)"
    }
    try:
        resp = requests.get(url, params=params, headers=headers, timeout=12)
        if resp.status_code != 200:
            return []
        data = resp.json()
        pages = data.get("query", {}).get("pages", {})
        urls = []
        for pid, page in pages.items():
            info = page.get("imageinfo", [])
            if info:
                # Prefer thumburl (640px resize) if available, else original url
                thumb = info[0].get("thumburl") or info[0].get("url")
                if thumb and not thumb.lower().endswith(('.svg', '.pdf', '.ogg', '.webm', '.tif', '.tiff')):
                    urls.append(thumb)
        return urls
    except Exception as e:
        print(f"  [API Error: {e}]", flush=True)
        return []

def download_and_validate(url, target_path, existing_hashes, timeout=10):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }
    try:
        resp = requests.get(url, headers=headers, timeout=timeout)
        if resp.status_code != 200:
            return False
        
        content = resp.content
        if len(content) < 3072: # Skip smaller than 3KB
            return False
            
        img_hash = hashlib.md5(content).hexdigest()
        if img_hash in existing_hashes:
            return False
            
        img = Image.open(BytesIO(content))
        img = img.convert("RGB")
        w, h = img.size
        if w < 120 or h < 120:
            return False
            
        img.save(target_path, "JPEG", quality=90)
        existing_hashes.add(img_hash)
        return True
    except Exception:
        return False

def collect_all():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("=== STARTING DATASET COLLECTION ===", flush=True)
    print(f"Classes: {list(CLASSES.keys())}", flush=True)
    print(f"Target per class: {TARGET_PER_CLASS}", flush=True)
    print(f"Output directory: {OUTPUT_DIR}\n", flush=True)
    
    total_downloaded = 0
    
    for cls_name, queries in CLASSES.items():
        cls_dir = os.path.join(OUTPUT_DIR, cls_name)
        os.makedirs(cls_dir, exist_ok=True)
        
        existing_files = [f for f in os.listdir(cls_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        print(f"--------------------------------------------------", flush=True)
        print(f"Class: [{cls_name.upper()}] (Current: {len(existing_files)} / Target: {TARGET_PER_CLASS})", flush=True)
        print(f"--------------------------------------------------", flush=True)
        
        existing_hashes = set()
        for f in existing_files:
            try:
                with open(os.path.join(cls_dir, f), "rb") as fp:
                    existing_hashes.add(hashlib.md5(fp.read()).hexdigest())
            except Exception:
                pass
                
        count = len(existing_files)
        if count >= TARGET_PER_CLASS:
            print(f"Class [{cls_name}] already reached target ({count} images). Skipping.\n", flush=True)
            continue
            
        for query in queries:
            if count >= TARGET_PER_CLASS:
                break
            print(f"  Searching: '{query}'...", flush=True)
            urls = fetch_wikimedia_images(query, max_results=50)
            print(f"  Found {len(urls)} candidate images.", flush=True)
            
            added = 0
            for u in urls:
                if count >= TARGET_PER_CLASS:
                    break
                filename = f"{cls_name}_{count+1:04d}.jpg"
                filepath = os.path.join(cls_dir, filename)
                if download_and_validate(u, filepath, existing_hashes):
                    count += 1
                    added += 1
                    total_downloaded += 1
                    if count % 10 == 0 or count == TARGET_PER_CLASS:
                        print(f"    Progress [{cls_name}]: {count}/{TARGET_PER_CLASS} images saved", flush=True)
                time.sleep(0.05)
                
            time.sleep(0.5)
            
        print(f"Completed [{cls_name}]: Total {count} images in dataset.\n", flush=True)
        
    print(f"=== DATASET COLLECTION FINISHED ===", flush=True)

if __name__ == "__main__":
    collect_all()
