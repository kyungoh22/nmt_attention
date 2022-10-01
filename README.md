This is Part 2 of my three-part Neural Machine Translation project. 

The overview of the project is as follows:
- Part 1: LSTMs 
- Part 2: Attention
- Part 3: Transformer

# Neural Machine Translation with Attention

### Intuition for the Attention mechanism
- A Neural Machine Translation model based on only LSTM units predicts a translation by using **a single state vector** to represent the entire source sentence all at once. 
- However, it is difficult to encode an entire sentence using just one state vector, particularly if we are dealing with a very long sentence. 
- The idea behind the Attention mechanism is to predict each word in the target sentence by paying attention to only a select number of words in the source sentence. 


## Model Architecture

### Encoder

- The Encoder comprises a series of bi-directional LSTM units. 
- The Encoder feeds the source sentence through an Embedding layer (in our case with pre-trained weights), 
before passing the embedding vectors through the LSTM units. 
- We retrieve the hidden state vectors **h<t>** from the LSTM units for every time-step. 
- This way, we have a state vector representation for every single word in the source sentence.

### Decoder

- The Decoder comprises an Attention layer and a series of LSTM units.
- The Attention layer outputs a **context vector** for every single time-step in the Decoder. 
- At each time-step, the Decoder concatenates the context vector from the current time-step with the Decoder output from the previous time-step (**s<t-1>**). 
- This concatenated vector is used as the input for the LSTM unit at the current time-step.

### Attention mechanism
- The context vector is a linear combination of all of the Encoder's hidden state vectors (**h<t>**).
– The weights of the state vectors represent how much the different words in the source sentence should be considered for the prediction at the current time-step. 
- The Attention layer is used to compute the weights. 
- To compute each weight, we concatenate the Decoder hidden state from the previous time-step (**s<t-1>**) with the Encoder hidden state from the current time-step (**h<t>**), before passing the concatenated vector through two dense layers.
- The weights are normalised using the softmax function.






## Training and Inference

### Training

We use teacher forcing to train our model. This means we compute the loss by comparing the ground-truth target output at time-step t with the prediction at time-step (t-1). 

### Inference

Once the model's weights have been optimized during the training, we can start making predictions. First, we pass the source sentence into our Encoder to obtain the final hidden state and final cell state. These two vectors alone represent the entire source sentence. The two state vectors are then passed into the Decoder as its initial states. The inference process takes place one time-step at a time. For each LSTM unit in the Decoder, we feed in the prediction from the previous time-step as the input, along with the hidden and cell states from the previous time-step. This iterative process continues for the pre-defined maximum number of time-steps, or until an LSTM unit predicts the "END" token.

## Remarks on tokenization
- I use word-based tokenization for this project. While this approach enables the use of pre-trained word embeddings, its major weakness is that the model can only work with words it has actually seen during the training. 
- In Part 3, I use BPE tokenization instead. As such, my Transformer model is much more robust when it comes to unseen words. 

## Data 
- The data for Part 1 comes from the website http://www.manythings.org/bilingual/
- The website collects sentence pairs for various language pairings. 
- For Part 1 of this project, I used English as the source language and German as the target language. 
- There are around 250,000 rows in the full dataset. I only used 10% of this data.

## Model parameters and performance
- I chose 256 as the length of each LSTM state vector.
- The pre-trained embeddings were downloaded from spaCy. The German embeddings have a length of 96; the English embedding a length of 300.
- I trained my model for 40 epochs, reaching a training accuracy of around 90% and validation accuracy of around 49%. 
- Since we have a high variance problem, training on the entire dataset would improve the validation accuracy. 



#### A note on the implementation of Part 1
I approached Part 1 as a kind of warm-up exercise for my project, knowing that the ultimate goal would be to implement the more complex Transformer architecture in Part 3. As such, all code for Part 1 has been written up in a single Jupyter notebook, entitled **LSTM_translator_clean.ipynb**. 

There are many instances where I could have written functions or custom sub-classes in a separate python file to make the code more elegant. For now, I left the code for Part 1 in its current, somewhat messy form. In Parts 2 and 3, I attempted to write my code in a much more structured manner.



