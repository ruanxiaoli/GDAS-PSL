class_dict = {
    'Nuclear': 0,
    'Cytoplasm': 1,
    'Vesicles': 2,
    'Mitochondria': 3,
    'Golgi apparatus': 4,
    'Endoplasmic reticulum': 5
}


def label_distribution(csv_file):
    with open(csv_file, 'r+') as f:
        lines = f.readlines()[1:]
    gene = []
    for line in lines:
        gene.append(line.replace("\n", '').split(",")[1])
    gene_key = list(set(gene))

    # 使用列表元素初始化字典，其值为一个空列表
    gene_dict = {key: [] for key in gene_key}
    for line in lines:
        gene = line.replace("\n", '').split(",")[1]
        categories = line.replace("\n", '').split(",")[-1].split(';')
        for category in categories:
            gene_dict[gene].append(category)
    label_distributions = [0, 0, 0, 0, 0, 0]
    for key in gene_key:
        gene_dict[key] = list(set(gene_dict[key]))
        for class_name in gene_dict[key]:
            label_distributions[class_dict[class_name]] += 1
    return label_distributions


if __name__ == "__main__":
    train_label = label_distribution('train.csv')
    test_label = label_distribution('test.csv')
    pass
