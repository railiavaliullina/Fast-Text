# Fast-Text

## About The Project
1) Implementation of FastText from 
paper: 

    https://arxiv.org/pdf/1607.04606.pdf
    
2) Training on AG News Classification Dataset for document classification task.


File to run:

    executor/executor.py

- After running executor.py, there is validation step on train and test data with the best checkpoint, and then training continues.

To run on Kaggle: 

    [FastText (final)](https://www.kaggle.com/code/rvnrvn1/fasttext-final)

## Additional Information

Visualization of accuracy on the training and test samples, loss are in: 

    saved_files/plots/

Confusion matrices are in: 

    saved_files/plots/Conf matrices/

MlFlow logs are in: 

    executor/mlruns.zip

Best achieved result:

    Fasttext bigram - accuracy: 87.65789473684211 %, test error: 12.34210526315789

    Fasttext no-bigram - accuracy: 87.02631578947368 %, test error: 12.973684210526315
