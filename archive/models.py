import tensorflow as tf
from tensorflow.keras import layers
from typing import Tuple
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Dropout
from tensorflow.keras.optimizers import RMSprop
import numpy as np

class OneLayer(tf.keras.Model):
    """Combined actor-critic network."""
    def __init__(self, 
                num_actions : int,
                num_hidden_units : int):
        super().__init__()

        self.common = layers.Dense(num_hidden_units, activation="relu")
        self.actor = layers.Dense(num_actions)
        self.critic = layers.Dense(1)

    def call(self, inputs: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        x = self.common(inputs)
        return self.actor(x), self.critic(x)

class a2c_Model:
    def __init__(self, observation_space, num_actions):

        input = Input(shape=(observation_space,),name='policy_input')
        
        #model's policy branch
        policy = Dense(32, activation="relu")(input)
        policy = Dense(16, activation="relu")(policy)
        logits = Dense(num_actions)(policy)

        #model's value function branch
        value_fn = Dense(32, activation="relu")(input)
        value_fn = Dense(16, activation="relu")(value_fn)
        value_fn = Dense(1)(value_fn)

        #defining the full model
        self.network = Model(inputs=input, outputs=[logits, value_fn])

    def forward_pass(self, inputs):
        #function to get forward pass
        x = tf.convert_to_tensor(inputs)
        return self.network(x)

    def actionfromdistribution(self, logits):
        #function to get a particular action from the logits of the different actions
        return tf.squeeze(tf.random.categorical(logits, 1), axis=-1)

    def action_value(self, observation):
        #function to return:
        #what action to take next
        #value function
        #based on input observation
        logits, value = self.forward_pass(observation)
        action = self.actionfromdistribution(logits)
        return np.squeeze(action, axis=-1), np.squeeze(value, axis=-1)
