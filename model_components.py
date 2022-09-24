import tensorflow as tf
from tensorflow.keras.initializers import Constant
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Concatenate, Dense #Bidirectional, Concatenate, LSTM, Embedding, Dense


class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, 
                    encoder_units, batch_size, embedding_matrix_source):
        super(Encoder, self).__init__()
        self.batch_size = batch_size
        self.encoder_units = encoder_units
        self.embedding = Embedding(vocab_size, embedding_dim, 
                                    embeddings_initializer = Constant(embedding_matrix_source),
                                    trainable = False)
        self.lstm= Bidirectional(LSTM (encoder_units, 
                                      return_sequences=True,
                                      return_state=True,                                      
                                      recurrent_initializer='glorot_uniform'
                                     ))
    def call(self, x):
                                                                # x = (m, Tx)
                                                                # hidden = (m, encoder_units)
                                                                
        """
        Computes Encoder's hidden state vectors for all time-steps, 
        as well as the final hidden and cell states. 
        Note, we are using bi-directional LSTMs,
        so the state vector's dimensions = 2 * encoder_units.

        Arguments
        x -- (m, Tx)

        Returns
        enc_sequential -- (m, Tx, 2 * encoder_units)
        enc_final_h -- (m, 2 * encoder_units)
        enc_final_c -- (m, 2 * encoder_units)
        """
        
        x = self.embedding(x)                                   # x = (m, Tx, embedding_dim)
        # pass input x through bi-directional LSTM
                                                                
        (enc_sequential, enc_forward_h, 
        enc_forward_c, enc_backward_h, enc_backward_c) = self.lstm(x)

        # concatenate forward and backward states
        enc_final_h = Concatenate()([enc_forward_h, enc_backward_h])
        enc_final_c = Concatenate()([enc_forward_c, enc_backward_c])

        return enc_sequential, enc_final_h, enc_final_c                     

    
class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super( BahdanauAttention, self).__init__()
        self.W1= tf.keras.layers.Dense(units)  # Dense layer for Decoder hidden state (at time-step "t-1")
        self.W2= tf.keras.layers.Dense(units)  # Dense layer for Encoder hidden state (at all time-steps, one at a time)
        self.V= tf.keras.layers.Dense(1)       # Dense layer to compute energy for each concatenated hidden state
    
    def call(self, dec_hidden, enc_hidden):
                                                                # dec_hidden = (m, 2*units) 
                                                                # enc_hidden = (m, Tx, 2*units)
        """
        Computes context vector using Encoder's hidden states and Decoder's hidden-state at time t-1
        
        Arguments
        dec_hidden -- decoder hidden state at t-1 -- (m, 2*units)
        enc_hidden -- encoder hidden states at all time-steps -- (m, Tx, 2*units)

        Returns
        context_vector -- (m, 2 * units)
        attention_weights -- (m, Tx, 1)
        """

        # expand dimensions to enable broadcasting
        dec_hidden_with_time = tf.expand_dims(dec_hidden, 1)    # dec_hidden_with_time = (m, 1, 2*units)
        
        # note on broadcasting:
        # W1(dec_hidden_with_time) = (m, 1, units)
        # W2 (enc_hidden) = (m, Tx, units)
        # Thanks to broadcasting:
        # W1 (dec_hidden_with_time) + W2 (enc_hidden) = (m, Tx, units)

        # Compute energy / score for each Encoder time-step
        # Linear + tanh activation + Linear 
        score = self.V(tf.nn.tanh(self.W1(dec_hidden_with_time) + self.W2(enc_hidden))) # (m, Tx, 1)
        
        # normalise scores with softmax
        attention_weights = tf.nn.softmax(score, axis=1)                                # (m, Tx, 1)
        
        # apply each weight to encoder hidden state at respective time-step 
        context_vector = attention_weights * enc_hidden                                  # (m, Tx, 2*units)
       
        # context_vector = linear combination of Encoder hidden state vectors for all Tx
        # So compute sum along Tx axis
        context_vector = tf.reduce_sum(context_vector, axis=1)                          # (m, 2*units)
        return context_vector, attention_weights



# Decoder for one time-step
class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, 
                    decoder_units, batch_sz, embedding_matrix_target, attention_layer_units):
        super (Decoder,self).__init__()
        self.batch_sz = batch_sz
        self.decoder_units = decoder_units
        self.embedding = Embedding(vocab_size, embedding_dim,
                                    embeddings_initializer = Constant(embedding_matrix_target),
                                    trainable = False)
        self.lstm= LSTM (decoder_units, 
                        return_sequences= True,
                        return_state=True,
                        recurrent_initializer='glorot_uniform')
        # Fully connected layer
        self.fc= Dense(vocab_size)      # Note, we don't use an activation here.
                                        # For the calculation of the loss, we will use 
                                        # sparse_softmax_cross_entropy_with_logits, which performs 
                                        # the softmax on the logits internally for greater efficiency
        
        # attention
        self.attention = BahdanauAttention(attention_layer_units)
    
    def call(self, y, dec_h, dec_c, enc_sequential):
                                                                                    # dec_h: (m, 2*units) 
                                                                                    # dec_c: (m, 2*units)
                                                                                    # enc_sequential: (m, Tx, 2*units) 
        """
        Computes the predicted y value, as well as the Decoder hidden and cell states,
        for current time-step t.

        Arguments
        y -- target input at time t-1 -- (m, 1)
        dec_h -- Decoder hidden state at time t-1 -- (m, 2*units)
        dec_c -- Decoder cell state at time t-1 -- (m, 2*units)
        enc_sequential -- Encoder hidden states at all time-steps -- (m, Tx, 2*units)

        Returns
        y -- target output at time t-1 -- ()
        """


        # compute context_vector and attention_weights using Attention layer
        context_vector, attention_weights = self.attention(dec_h, enc_sequential)   # context_vector = (m, 2*units)
        
        # compute embedding for previous time-step of output 
        y = self.embedding(y)                                                        # y = (m, 1, embedding_dim)
        
        # concatenate context vector and embedding for output sequence
        y = tf.concat([tf.expand_dims(context_vector, 1), y],                       # (m, 1, 2*units) + (m, 1, embedding_dim)
                                      axis = -1)                                    # (m, 1, 2*units + embedding_dim)
        
        # passing the concatenated vector to the LSTM
        output, dec_h, dec_c = self.lstm(y, initial_state = [dec_h, dec_c])         # output = (m, 1, 2*units)
                                                                                    # dec_h = (m, 2*units)
                                                                                    # dec_c = (m, 2*units)

        output = tf.reshape(output, (-1, output.shape[2]))                          # output = (m, 2*units)
        
        # pass the output through Dense layer to get logits
        y = self.fc(output)                                                         # y = (m, vocab_size)
        return y, dec_h, dec_c, attention_weights