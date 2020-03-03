import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv('skindataall.csv', index_col=[0])

# df.columns

combined = df[['Product', 'User_id', 'Rating_Stars']]

# print('Number of unique products: ', combined['Product'].nunique())
# print('Number of unique users: ', combined['User_id'].nunique())

scaler = MinMaxScaler()
combined['Rating_Stars'] = combined['Rating_Stars'].values.astype(float)
rating_scaled = pd.DataFrame(scaler.fit_transform(combined['Rating_Stars'].values.reshape(-1,1)))
combined['Rating_Stars'] = rating_scaled

combined = combined.drop_duplicates(['User_id', 'Product'])
user_product_matrix = combined.pivot(index='User_id', columns='Product', values='Rating_Stars')
user_product_matrix.fillna(0, inplace=True)

users = user_product_matrix.index.tolist()
products = user_product_matrix.columns.tolist()

user_product_matrix = user_product_matrix.as_matrix()

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

num_input = combined['Product'].nunique()
num_hidden_1 = 10
num_hidden_2 = 5

X = tf.placeholder(tf.float64, [None, num_input])

weights = {
    'encoder_h1': tf.Variable(tf.random_normal([num_input, num_hidden_1], dtype=tf.float64)),
    'encoder_h2': tf.Variable(tf.random_normal([num_hidden_1, num_hidden_2], dtype=tf.float64)),
    'decoder_h1': tf.Variable(tf.random_normal([num_hidden_2, num_hidden_1], dtype=tf.float64)),
    'decoder_h2': tf.Variable(tf.random_normal([num_hidden_1, num_input], dtype=tf.float64)),
}

biases = {
    'encoder_b1': tf.Variable(tf.random_normal([num_hidden_1], dtype=tf.float64)),
    'encoder_b2': tf.Variable(tf.random_normal([num_hidden_2], dtype=tf.float64)),
    'decoder_b1': tf.Variable(tf.random_normal([num_hidden_1], dtype=tf.float64)),
    'decoder_b2': tf.Variable(tf.random_normal([num_input], dtype=tf.float64)),
}

def encoder(x):
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']), biases['encoder_b1']))
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']), biases['encoder_b2']))
    return layer_2

def decoder(x):
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']), biases['decoder_b1']))
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']), biases['decoder_b2']))
    return layer_2

encoder_op = encoder(X)
decoder_op = decoder(encoder_op)

y_pred = decoder_op

y_true = X

loss = tf.losses.mean_squared_error(y_true, y_pred)
optimizer = tf.train.RMSPropOptimizer(0.03).minimize(loss)
eval_x = tf.placeholder(tf.int32, )
eval_y = tf.placeholder(tf.int32, )
pre, pre_op = tf.metrics.precision(labels=eval_x, predictions=eval_y)

init = tf.global_variables_initializer()
local_init = tf.local_variables_initializer()
pred_data = pd.DataFrame()

with tf.Session() as session:
    epochs = 50
    batch_size = 35

    session.run(init)
    session.run(local_init)

    num_batches = int(user_product_matrix.shape[0] / batch_size)
    user_product_matrix = np.array_split(user_product_matrix, num_batches)

    for i in range(epochs):

        avg_cost = 0
        for batch in user_product_matrix:
            _, l = session.run([optimizer, loss], feed_dict={X: batch})
            avg_cost += l

        avg_cost /= num_batches

        print("epoch: {} Loss: {}".format(i + 1, avg_cost))

    user_product_matrix = np.concatenate(user_product_matrix, axis=0)

    preds = session.run(decoder_op, feed_dict={X: user_product_matrix})

    pred_data = pred_data.append(pd.DataFrame(preds))

    pred_data = pred_data.stack().reset_index(name='Rating_Stars')
    pred_data.columns = ['User_id', 'Product', 'Rating_Stars']
    pred_data['User_id'] = pred_data['User_id'].map(lambda value: users[value])
    pred_data['Product'] = pred_data['Product'].map(lambda value: products[value])

    keys = ['User_id', 'Product']
    index_1 = pred_data.set_index(keys).index
    index_2 = combined.set_index(keys).index

    top_ten_ranked = pred_data[~index_1.isin(index_2)]
    top_ten_ranked = top_ten_ranked.sort_values(['User_id', 'Product'], ascending=[True, False])
    top_ten_ranked = top_ten_ranked.groupby('User_id').head(10)


top_ten_ranked.loc[top_ten_ranked['User_id'] == 6598]


combined['User_id'].value_counts()
