import datasets


def code_anchor(example):
    whole_func_string = str(example['content'])
    example['anchor'] = whole_func_string.replace('\n', '\n<ANCHOR>').replace("    ", "\t")
    example['content'] = None

    return example

if __name__ == '__main__':

    dataset = datasets.load_from_disk("your_dataset")
    dataset = dataset.map(code_anchor)
    dataset.save_to_disk('anchored_dataset')