import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def plot_label_distribution(labels, split, class_names):
    """
    Plots a normalized frequency distribution of classes for a given dataset split.
    Helps identify class imbalance issues as required by Task 1a.
    """
    # Map integer labels back to explicit textual class names
    mapped_labels = [class_names[idx] for idx in labels]
    df = pd.DataFrame({'Lesion Type': mapped_labels})
    
    plt.figure(figsize=(9, 4))
    sns.countplot(
        data=df, 
        x='Lesion Type', 
        order=class_names, 
        palette='magma', 
        hue='Lesion Type', 
        legend=False
    )
    
    plt.title(f'{split} Dataset - Class Distribution Matrix')
    plt.ylabel('Total Sample Count')
    plt.xlabel('Diagnostic Categories')
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()