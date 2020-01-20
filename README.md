# NLP-Topic-Recognition
The main aim of the program is to classify into general categories the dataset with conversations labelled by more 
specialised topics. 
The code implies the classification task trained on the train.csv dataset, tested on valid.csv dataset and used for 
bridged_10k.csv. Last dataset differs from the previous two containing the conversations instead of normal sentences, so 
all the phrases were classified and major category was chosen to the specialised topics. 
As training data appears to be highly unbalanced, several topics were combined('Emotional abuse', 'Suicide and SelfHarm'
 => 'Emotion', 'Sexual abuse' => 'Sex', 'Physical abuse' => 'Health', 'Terrorism / extremism' => 'Politics'), also the 
overloaded categories were limited to 5000 words each and for the categories containing less samples, the data was 
increased to this number of samples by smuto algotithm implemented on vectorized data. Glove vectorization was used on 
the tokenized words averaging for the sentences with removing stop_words. 
The dimensionality reduction of word vectors was implemented by the NMF.
For the classification the Support Vector Classification was used and the topics that did not get the probabilities more 
than 0.1 were moved to 'Others'. 

The input variables are: 
number of components for feature selection(max 300)
number of samples per categories used for training
threshold probability to regulate the number of classes as output
