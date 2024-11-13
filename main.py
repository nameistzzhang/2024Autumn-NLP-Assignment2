from dataHelper import get_dataset

DATASET_NAME = {
    "laptop_sup",
    "restaurant_sup",
    "acl_sup", 
    "agnews_sup",
    "laptop_fs",
    "restaurant_fs",
    "acl_fs",
    "agnews_fs"
}

dataset_name = ["agnews_sup"]
sep_token = "<sep>"
dataset = get_dataset(dataset_name, sep_token)
