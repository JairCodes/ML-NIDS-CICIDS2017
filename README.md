# Machine-Learning NIDS with CIC-IDS-2017

## Project Overview
This project implements a machine-learning-based Network Intrusion Detection System (NIDS) using the CIC-IDS-2017 dataset.  
**Goals:**
- Train and compare three classifiers (Random Forest, SVM, Deep Neural Network) on flow-based features.
- Evaluate trade-offs in accuracy, recall, precision and F1 for underrepresented attack types.
- Demonstrate preprocessing, feature engineering, and class-balancing techniques.

---

## Dataset Download & Installation

1. Go to the CIC-IDS-2017 page:  
   👉 http://cicresearch.ca/CICDataset/CIC-IDS-2017/  
2. Fill out the “CIC Dataset Download Form” with your First/Last name, email, etc.  
3. Click the **CSVs/** directory.  
4. Download **MachineLearningCSV.zip**.  
5. Unzip and place the resulting folder in your repo root as:
   ```text
   your-repo/
   └── data/
       └── MachineLearningCSV/
