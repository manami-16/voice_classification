# Emotional Classifier Using Audio Data
### Physics 4606 Final Project at Northeastern University <br>
### Manami Kanemura 

The goal of this project is to study the signal processing for the human voice and to build deep learning models to classify a speaker’s emotion based on voice data (angry or happy). The differences in signal preprocessing between how computers detect our voice and how humans detect our voice by applying Fast Fourier Transform (FFT) and Mel-scale spectrums are determined. After the analysis of voice data, convolutional neural networks (CNN) with different pooling layers are built to classify the binary emotion from vocal data. After training 80 - 100 epochs, CNN with MaxPool1d scored 0.913 AUC whereas CNN with AvgPool1d measured 0.645 AUC. 