# ML-Projects
Collection of various ML scripts/notebooks (mostly neural nets in pytorch) trained on kaggle datasets.  

## Dogs
Files: <code>dogs.ipynb</code> & <code>dogs.py</code><br>
Trained a dog breed classifier on the Stanford dogs dataset using a pytorch convolutional neural net. Experimented with image size (64x64 or 128x128 pixels), network architecture (# of convolutional layers, dropout probabilities), and training hyperparameters. 

## Corona X-Ray Dataset
Files: <code>corona.ipynb</code> & <code>corona.py</code><br>
Kaggle dataset of ~5,000 chest x-rays of 3 types: normal lungs, bacterial pnemonia, and viral pnemonia (including SARS-COV-1 & SARS-COV-2.) I train a pytorch multi-class convolutional neural net to predict on a 256x256 image.

## MNIST
Files <code>mnist.ipynb</code> & <code>mnist gan.ipynb</code><br>
The first notebook is the "hello world" of image machine learning -- using a pytorch neural net to classify handwritten digits from the MNIST dataset.
The other notebook is experimenting with GANs in TF/keras. 

## Credit Cards
Files: <code>credit cards.ipynb</code><br>
Kaggle dataset of anonymized credit card transactions classified as legitimate or fraud. Trained an XGBoost binary classifier.
