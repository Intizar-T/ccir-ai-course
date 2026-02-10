
import sys
import os

# Add the directory containing train.py to the python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from train import process_data

def test_process_data():
    try:
        df = process_data()
        print("Data loaded successfully!")
        print(f"Shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        print(f"First 5 rows:\n{df.head()}")
        
        if 'Date' in df.columns:
            print(f"Date column type: {df['Date'].dtype}")
            
    except Exception as e:
        print(f"Error processing data: {e}")

if __name__ == "__main__":
    test_process_data()
