import json

def load_groundtruth():
    with open("data/evaluation_dataset/groundtruth.json") as f:
        return json.load(f)

gt = load_groundtruth()
print(f"Loaded {len(gt)} groundtruth records")
for i, item in enumerate(gt[:2]):
    print(f"[{i}] {item.get('filename')}:")
    print(f"    text len: {len(item.get('groundtruth_text', ''))}")
    
