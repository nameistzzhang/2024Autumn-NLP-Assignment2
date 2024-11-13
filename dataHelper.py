import json
from datasets import Dataset, DatasetDict
from datasets import load_dataset

def processABSA(dataset_name, sep_token, fewshot=False):
    """
	sep_token: str, the sep_token used by tokenizer(e.g. '<sep>')
	return: a DatasetDict object
	"""
    dataset = None
    trainset_path = "./data/SemEval14-laptop/train.json"
    testset_path = "./data/SemEval14-laptop/test.json"
    if dataset_name == "restaurant_sup":
        trainset_path = "./data/SemEval14-res/train.json"
        testset_path = "./data/SemEval14-res/test.json"
	
    with open(trainset_path, 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    with open(testset_path, 'r', encoding='utf-8') as f:
        test_data = json.load(f)

    """
    The structure of this dataset is:
    dict{"entry_key": dict{"polarity": str, "term": str, "id": str, "sentence": str}}
    """

    train_texts = []
    train_labels = []
    test_texts = []
    test_labels = []
    # use sentiment_map.get(polarity, -1) for the case when polarity is not in sentiment_map
    sentiment_map = {"positive": 0, "neutral": 1, "negative": 2}

    for entry in train_data.values():
        term = entry['term']
        sentence = entry['sentence']
        polarity = entry['polarity']
        
        text = f"{term} {sep_token} {sentence}"
        train_texts.append(text)
        train_labels.append(sentiment_map.get(polarity, -1))
    if fewshot:
        # only use a few samples for training
        train_texts = train_texts[:32]
        train_labels = train_labels[:32]

    for entry in test_data.values():
        term = entry['term']
        sentence = entry['sentence']
        polarity = entry['polarity']
        
        text = f"{term} {sep_token} {sentence}"
        test_texts.append(text)
        test_labels.append(sentiment_map.get(polarity, -1))

    train_dataset = Dataset.from_dict({"text": train_texts, "label": train_labels})
    test_dataset = Dataset.from_dict({"text": test_texts, "label": test_labels})

    dataset = DatasetDict({"train": train_dataset, "test": test_dataset})

    return dataset

def processACL(dataset_name, sep_token, fewshot=False):
    """
	sep_token: str, the sep_token used by tokenizer(e.g. '<sep>')
	return: a DatasetDict object
	"""
    dataset = None
    trainset_path = "./data/acl_sup/train.json"
    testset_path = "./data/acl_sup/test.json"

    # the json structure
    """
    line1: dict{'text': str, 'label': str, 'metadata': str}
    line2: dict{'text': str, 'label': str, 'metadata': str}
    ...
    """

    with open(trainset_path, 'r') as file:
        train_data = [json.loads(line) for line in file]  # 逐行读取 JSON
    with open(testset_path, 'r') as file:
        test_data = [json.loads(line) for line in file]

    sentiment_map = {'CompareOrContrast': 0, 'Extends': 1, 'Uses': 2, 'Background': 3, 'Future': 4, 'Motivation': 5}
    train_texts = []
    train_labels = []
    test_texts = []
    test_labels = []
    for entry in train_data:
        text = entry['text']
        label = entry['label']
        train_texts.append(text)
        train_labels.append(sentiment_map.get(label, -1))
    if fewshot:
        # only use a few samples for training
        train_texts = train_texts[:32]
        train_labels = train_labels[:32]
    for entry in test_data:
        text = entry['text']
        label = entry['label']
        test_texts.append(text)
        test_labels.append(sentiment_map.get(label, -1))

    train_dataset = Dataset.from_dict({"text": train_texts, "label": train_labels})
    test_dataset = Dataset.from_dict({"text": test_texts, "label": test_labels})

    dataset = DatasetDict({"train": train_dataset, "test": test_dataset})

    return dataset

def processAGNEWS(dataset_name, sep_token, fewshot=False):
    """
	sep_token: str, the sep_token used by tokenizer(e.g. '<sep>')
	return: a DatasetDict object
	"""
    dataset = load_dataset('wangrongsheng/ag_news', split="test")
    
    # use the train_test_split method to split the dataset into train and test (9:1) and put it in a DatasetDict object
    dataset = dataset.train_test_split(test_size=0.1, seed=2022)

    # if fewshot
    if fewshot:
        # only use a few samples for training
        dataset["train"] = dataset["train"].select(list(range(32)))

    return dataset

def str_dataset(dataset_name, sep_token):

    if dataset_name == "laptop_sup" or dataset_name == "restaurant_sup":
        return processABSA(dataset_name, sep_token)
    elif dataset_name == "acl_sup":
        return processACL(dataset_name, sep_token)
    elif dataset_name == "agnews_sup":
        return processAGNEWS(dataset_name, sep_token)
    elif dataset_name == "laptop_fs" or dataset_name == "restaurant_fs":
        return processABSA(dataset_name, sep_token, fewshot=True)
    elif dataset_name == "acl_fs":
        return processACL(dataset_name, sep_token, fewshot=True)
    elif dataset_name == "agnews_fs":
        return processAGNEWS(dataset_name, sep_token, fewshot=True)
    else:
        raise ValueError("Invalid dataset name")

def get_dataset(dataset_name, sep_token):
    """
    dataset_name: str, the name of the dataset
    sep_token: str, the sep_token used by tokenizer(e.g. '<sep>')
    return: a DatasetDict object
    """

    if type(dataset_name) == str:
        dataset_name = [dataset_name]
    elif type(dataset_name) == list:
        dataset_name = dataset_name
    else:
        raise ValueError("dataset_name should be a string or a list of strings")

    datasets = []
    for name in dataset_name:
        datasets.append(str_dataset(name, sep_token))
    
    if len(datasets) == 1:
        return datasets[0]

    # Initialize lists to hold concatenated texts and labels
    all_train_texts = []
    all_train_labels = []
    all_test_texts = []
    all_test_labels = []

    # Offset for relabeling to avoid label overlapping
    label_offset = 0

    for dataset in datasets:
        # Relabel train labels
        train_labels = [label + label_offset for label in dataset["train"]["label"]]
        all_train_texts.extend(dataset["train"]["text"])
        all_train_labels.extend(train_labels)

        # Relabel test labels
        test_labels = [label + label_offset for label in dataset["test"]["label"]]
        all_test_texts.extend(dataset["test"]["text"])
        all_test_labels.extend(test_labels)

        # Update label offset
        label_offset = max(train_labels + test_labels) + 1

    # Create concatenated train and test datasets
    concatenated_train_dataset = Dataset.from_dict({"text": all_train_texts, "label": all_train_labels})
    concatenated_test_dataset = Dataset.from_dict({"text": all_test_texts, "label": all_test_labels})

    # Create final DatasetDict
    concatenated_dataset = DatasetDict({"train": concatenated_train_dataset, "test": concatenated_test_dataset})

    return concatenated_dataset
