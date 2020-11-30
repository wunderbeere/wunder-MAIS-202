import json

artists_info_dict = json.load(open(".\\static\\artists_info.json"))

print(artists_info_dict["Vincent van Gogh"]["biography"])
