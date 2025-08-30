import os
import pandas as pd

# Get the list of existing genes in the file system
data_dir = "../ndata/xiasina/hpa"
gene_reliable = ["enhanced"]
intensity = "inensity_level_strong"

existing_genes = set()
for i in gene_reliable:
    gene_dir = os.path.join(data_dir, i, intensity)
    if os.path.exists(gene_dir):
        for item in os.listdir(gene_dir):
            if os.path.isdir(os.path.join(gene_dir, item)):
                existing_genes.add(item)

# Read genes from CSV files
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")
needed_genes = set(train_df['gene'].unique()) | set(test_df['gene'].unique())

# Find missing genes
missing_genes = needed_genes - existing_genes

print("\nMissing genes:")
for gene in missing_genes:
    print(gene)

print(f"\nTotal number of missing genes: {len(missing_genes)}")

# Save missing gene information to file
with open('missing_genes.txt', 'w') as f:
    for gene in missing_genes:
        f.write(gene + '\n')

print("\nMissing genes have been saved to 'missing_genes.txt'")