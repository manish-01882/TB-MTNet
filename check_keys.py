import json
with open("Annotations_AllinOne_json.json", "r") as f:
    data = json.load(f)
keys = set()
for entry in data.values():
    for region in entry.get("regions", []):
        for k in region.get("region_attributes", {}).keys():
            keys.add(k)
print("Keys in region_attributes:", keys)
