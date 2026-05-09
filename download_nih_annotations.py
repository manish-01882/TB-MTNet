import requests
from bs4 import BeautifulSoup
import os
import shutil
from pathlib import Path
from urllib.parse import urljoin

BASE_URL = "https://data.lhncbc.nlm.nih.gov/public/Tuberculosis-Chest-X-ray-Datasets/Shenzhen-Hospital-CXR-Set/Annotations/masks/index.html"
OUTPUT_DIR = Path("Masks_nih")
LOCAL_SOURCE_DIR = Path(__file__).resolve().parent / "mask"
OUTPUT_DIR.mkdir(exist_ok=True)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

def download_file(url, dest_path):
    print(f"  downloading {dest_path.name} …", end=" ")
    r = requests.get(url, headers=HEADERS)
    if r.status_code == 200:
        dest_path.write_bytes(r.content)
        print("ok")
    else:
        print(f"FAILED ({r.status_code})")


def copy_local_folder(source_folder, local_folder):
    local_folder.mkdir(exist_ok=True)
    print(f"\nCopying from local source {source_folder}")
    for source_path in sorted(source_folder.glob("*.png")):
        dest_path = local_folder / source_path.name
        print(f"  copying {dest_path.name} …", end=" ")
        shutil.copy2(source_path, dest_path)
        print("ok")

def download_folder(url, local_folder):
    local_folder.mkdir(exist_ok=True)
    print(f"\nEntering {url}")
    r = requests.get(url, headers=HEADERS)
    if r.status_code != 200:
        print(f"Cannot access {url}")
        if LOCAL_SOURCE_DIR.exists():
            copy_local_folder(LOCAL_SOURCE_DIR, local_folder)
        else:
            print(f"Local fallback source not found: {LOCAL_SOURCE_DIR}")
        return
    soup = BeautifulSoup(r.text, "html.parser")
    for link in soup.find_all("a"):
        name = link.get("href")
        if not name or name.startswith("?") or name.startswith("/") or name.startswith("../"):
            continue
        if name.lower().endswith(".png"):
            download_file(urljoin(url, name), local_folder / name)
        elif name.endswith("/"):   # subfolder
            download_folder(urljoin(url, name), local_folder / name.rstrip("/"))
        else:
            continue

download_folder(BASE_URL, OUTPUT_DIR)
print("\nDone.")