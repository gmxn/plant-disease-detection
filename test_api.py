import requests
import json
import sys
from pathlib import Path

BASE_URL = "http://127.0.0.1:8000"

def test_root():
    print("Testing GET / ...")
    r = requests.get(f"{BASE_URL}/")
    print(f"  Status: {r.status_code}")
    print(f"  Response: {json.dumps(r.json(), indent=2)}\n")

def test_classes():
    print("Testing GET /classes ...")
    r = requests.get(f"{BASE_URL}/classes")
    data = r.json()
    print(f"  Status: {r.status_code}")
    print(f"  Total classes: {data['total']}")
    print(f"  First 3: {data['classes'][:3]}\n")

def test_predict(image_path: str):
    print(f"Testing POST /predict with {image_path} ...")
    with open(image_path, "rb") as f:
        r = requests.post(
            f"{BASE_URL}/predict",
            files={"file": (Path(image_path).name, f, "image/jpeg")}
        )
    print(f"  Status: {r.status_code}")
    data = r.json()
    print(f"  Prediction: {data['prediction']['crop']} — {data['prediction']['disease']}")
    print(f"  Confidence: {data['prediction']['confidence']}%")
    print(f"  Top 3 predictions:")
    for p in data['top_3']:
        print(f"    #{p['rank']} {p['crop']} — {p['disease']}: {p['confidence']}%")

if __name__ == "__main__":
    test_root()
    test_classes()
    if len(sys.argv) > 1:
        test_predict(sys.argv[1])
    else:
        print("To test prediction, run:")
        print("  python test_api.py path\\to\\leaf_image.jpg")