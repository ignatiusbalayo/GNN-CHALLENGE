import pandas as pd
from sklearn.model_selection import train_test_split

def create_imbalanced_split(input_csv, train_size=0.1):
    df = pd.read_csv(input_csv)
    
    # Stratified split ensures all classes are represented even with imbalance
    train, temp = train_test_split(
        df, train_size=train_size, stratify=df['label'], random_state=42
    )
    
    # Split remaining into Val and Test
    val, test = train_test_split(
        temp, train_size=0.5, stratify=temp['label'], random_state=42
    )
    
    # Action: MODIFY - Save these to your data/ folder
    train.to_csv('data/train.csv', index=False)
    val.to_csv('data/val.csv', index=False)
    
    # Remove labels for the public test set
    test_no_label = test.drop(columns=['label'])
    test_no_label.to_csv('data/test_no_label.csv', index=False)
    
    # Keep this SECURE and NOT in GitHub (used by evaluate.py via secret)
    test.to_csv('data/private_labels.csv', index=False)

if __name__ == "__main__":
    create_imbalanced_split('raw_citeseer_data.csv')