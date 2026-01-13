import json
import yaml

def read_json(json_path: str):
    with open(json_path, "r", encoding="utf-8") as file:
        return json.load(file)
    

def write_yaml(yaml_path: str, data: dict):
    with open(yaml_path, "w", encoding="utf-8") as file:
        yaml.dump(data, file, sort_keys=False)

def write_txt(txt_path: str, content: str):
    with open(txt_path, "w", encoding="utf-8") as file:
        file.write(content)