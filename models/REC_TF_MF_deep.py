import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from keras.models import load_model
from keras.layers import BatchNormalization, Dropout
from keras.models import Model
from keras.layers import Input, Embedding, Dot, Add, Flatten
from keras.regularizers import l2
from keras.optimizers import SGD, Adam
import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, Dot, Add, Flatten, Concatenate, Dense, Activation
from tensorflow.keras.models import Model
import mlflow

class REC_TF_MF_deep:

    def __init__(self,df=None):
        if df is not None:
            self.df = df
            self.mu = 0.5
            self.K = 10
            self.reg = 0.0001
            self.epochs = 1
            self.__model_init(K=10,reg=0.0001)
            print('model__created', self.model)
        else:
            print('model_created')


    def __model_init(self,K,reg):
        N = int(self.df.encoded_userId.max() + 1)  # number of users
        M = int(self.df.encoded_receiverId.max() + 1)  # number of movies
        print('model initing...')
        with tf.device('/CPU:0'):
            u = Input(shape=(1,))
            m = Input(shape=(1,))

            u_embedding = Embedding(N, K, embeddings_regularizer=l2(reg))(u)
            m_embedding = Embedding(M, K, embeddings_regularizer=l2(reg))(m)
            u_embedding = Flatten()(u_embedding)
            m_embedding = Flatten()(m_embedding)
            x = Concatenate()([u_embedding, m_embedding])  # (N, 2K)

            # the neural network
            x = Dense(512)(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            x = Dropout(0.5)(x)
            x = Dense(256)(x)
            x = BatchNormalization()(x)
            x = Dropout(0.5)(x)
            x = Dense(128)(x)
            x = BatchNormalization()(x)
            x = Dropout(0.5)(x)
            x = Dense(64)(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            x = Dense(1)(x)

        model = Model(inputs=[u, m], outputs=x)
        model.compile(
            loss='mae',
            # optimizer='adam',
            optimizer=Adam(lr=0.001),
            # optimizer=SGD(lr=0.08, momentum=0.9),
            metrics=['mse', 'mae'],
        )
        print('model init')
        self.model = model

    def train_model(self,train,K,reg,epochs):

        self.df = shuffle(self.df)
        self.K=K
        self.epochs = epochs
        self.reg = reg
        cutoff = int(train * len(self.df))
        df_train = self.df.iloc[:cutoff]
        df_test = self.df.iloc[cutoff:]
        self.mu = df_train.rating.mean()


        self.__model_init(K,reg)

        mlflow.autolog()
        self.rait = self.model.fit(
            x=[df_train.encoded_userId.values, df_train.encoded_receiverId.values],
            y=df_train.rating.values - self.mu,
            epochs=epochs,
            batch_size=128,
            validation_data=(
                [df_test.encoded_userId.values, df_test.encoded_receiverId.values],
                df_test.rating.values - self.mu
            )
        )
        self.save_model('models/last_model.h5')
    def save_model(self,path):
        self.model.save(path)

    def load_model(self,path):
        del self.model
        self.model = load_model(path)

    def predict(self,input):
        if self.mu is None:
            self.mu = .5

        return self.model.predict(input) + self.mu
    def update_model(self,dataset,train=0.8, K=10, reg=0.0001, epochs=1):
        merged_df = pd.concat([self.df[['encoded_userId', 'encoded_receiverId', 'rating']],
                               dataset[['encoded_userId', 'encoded_receiverId', 'rating']]], ignore_index=True)
        merged_df.groupby(by=['encoded_userId', 'encoded_receiverId']).mean().reset_index()

        print('DATASET SHAPE: ',self.df.shape)
        #mu = self.df.rating.mean()
        #x = [self.df.encoded_userId.values, self.df.encoded_receiverId.values],
        #y = self.df.rating.values - mu,
        self.df = merged_df
        self.train_model(train, K, reg, epochs=epochs)


    def concept_update(self,dataset,train=0.8, K=10, reg=0.0001, epochs=1):

        merged_df = pd.concat(
            [self.df[['encoded_userId', 'encoded_receiverId', 'rating']],
                               dataset[['encoded_userId', 'encoded_receiverId', 'rating']]], ignore_index=True)
        merged_df = merged_df.groupby(by=['encoded_userId', 'encoded_receiverId']).mean().reset_index()

        print('DATASET SHAPE: ', self.df.shape)
        # mu = self.df.rating.mean()
        # x = [self.df.encoded_userId.values, self.df.encoded_receiverId.values],
        # y = self.df.rating.values - mu,
        self.df = dataset
        # Define the new value for N (number of users)
        N = int(self.df.encoded_userId.max() + 1)
        M = int(self.df.encoded_receiverId.max() + 1)

        u = Input(shape=(1,))
        m = Input(shape=(1,))
        user_embedding_layer = Embedding(N, self.K,embeddings_regularizer=l2(reg))(u)
        item_embedding_layer = Embedding(M, self.K,embeddings_regularizer=l2(reg))(m)

        # Find the indices of the user and item embedding layers in the model
        index_of_user_embedding_layer = -1
        index_of_item_embedding_layer = -1
        for i, layer in enumerate(self.model.layers):
            if layer.name == 'user_embedding':
                index_of_user_embedding_layer = i
            elif layer.name == 'item_embedding':
                index_of_item_embedding_layer = i

        # Replace the embedding layers in the original model
        self.model.layers[index_of_user_embedding_layer] = user_embedding_layer
        self.model.layers[index_of_item_embedding_layer] = item_embedding_layer

        # Compile the model with the appropriate loss function and optimizer
        self.model.compile(loss='mae', optimizer='adam')

        # Train the model on the new data with the updated N value
        # You can use the same training data (new_data and new_targets) as mentioned earlier
        cutoff = int(train * len(self.df))
        df_train = self.df.iloc[:cutoff]
        df_test = self.df.iloc[cutoff:]
        self.mu = df_train.rating.mean()



        #mlflow.autolog()
        self.rait = self.model.fit(
            x=[df_train.encoded_userId.values, df_train.encoded_receiverId.values],
            y=df_train.rating.values - self.mu,
            epochs=epochs,
            batch_size=128,
            validation_data=(
                [df_test.encoded_userId.values, df_test.encoded_receiverId.values],
                df_test.rating.values - self.mu
            )
        )
        self.save_model('models/last_model.h5')





