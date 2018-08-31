import sys
import argparse
import multiprocessing as mp
import numpy as np
import tensorflow as tf
#from tensorflow.layers.embeddings import Embedding
#from tensorflow.keras.layers import Input, Embedding, l2, Dense
#from tensorflow.keras.models import Model
from Dataset import Dataset
from time import time

#################### Arguments #################333
def parse_args():
    parser = argparse.ArgumentParser(description="Run MLP")
    parser.add_argument('--path', nargs='?', default='Data/',
                        help='Input data path.')
    parser.add_argument('--dataset', nargs='?', default='ml-1m',
                        help='Choose a dataset')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs.')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size.')
    parser.add_argument('--layers', nargs='?', default='[64,32,16,8]',
                        help="Size of each layer. Note that the first layer is the concatenation of user and item embeddings. So layers[0]/2 is the embedding size.")
    parser.add_argument('--reg_layers', nargs='?', default='[0,0,0,0]',
                        help="Regularization for each layer")
    parser.add_argument('--num_neg', type=int, default=4,
                        help='Number of negative instances to pair with a positive instance.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate.')
    parser.add_argument('--learner', nargs='?', default='adam',
                        help='Specify an optimizer: adagrad, adam, rmsprop, sgd')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Show performance per X iterations')
    parser.add_argument('--out', type=int, default=1,
                        help='Whether to save the trained model.')
    parser.add_argument('--allow_soft_placement', type=bool, default=True,
                        help='Allow device soft device placement')
    parser.add_argument('--log_device_placement', type=bool, default=False,
                        help='Log placement of ops on devices')
    return parser.parse_args()

def init_normal(shape, name=None):
    return "glorot_uniform"

class MLP:
    def __init__(self, num_users, num_items, layers=[20,10], reg_layers=[0,0]):
        #def get_model(num_users, num_items, layers=[20,10], reg_layers=[0,0]):
        assert len(layers) == len(reg_layers)
        num_layer = len(layers) # Number of layers in the MLP

        # Input variables
        #user_input = tf.keras.layers.Input(shape=(1,), dtype='int32', name='user_input')
        #item_input = tf.keras.layers.Input(shape=(1,), dtype='int32', name='item_input')
        l2_loss = tf.constant(0.0)
        self.user_input = tf.placeholder(tf.int32, shape=[None,1], name='user_input') # check shape
        self.item_input = tf.placeholder(tf.int32, shape=[None,1], name='item_input') # check shape

        '''
        MLP_Embedding_User = tf.keras.layers.Embedding(num_users,layers[0]/2, name='user_embedding',
                                init=init_normal, W_regularizer=l2(reg_layers[0]), input_length=1)
        MLP_Embedding_Item = tf.keras.layers.Embedding(num_items, layers[0]/2, name='item_embedding',
                                init=init_normal, W_regularizer=l2(reg_layers[0]), input_length=1)
        '''
        with tf.device('/gpu:0'), tf.name_scope('user_embedding'):
            self.W_user = tf.Variable(tf.random_uniform([num_users, int(layers[0]/2)], -1.0, 1.0), name='W_user')
            MLP_Embedding_User = tf.nn.embedding_lookup(self.W_user, self.user_input)
        with tf.device('/gpu:0'), tf.name_scope('item_embedding'):
            #print(item_input.shape)
            self.W_item = tf.Variable(tf.random_uniform([num_items, int(layers[0]/2)], -1.0, 1.0), name='W_item')
            MLP_Embedding_Item = tf.nn.embedding_lookup(self.W_item, self.item_input)
            #print(MLP_Embedding_Item.shape)

        # Crucial to flatten an embedding_vector
        self.user_latent = tf.layers.Flatten()(MLP_Embedding_User)
        self.item_latent = tf.layers.Flatten()(MLP_Embedding_Item)
        print(self.user_latent.shape)
        print(self.item_latent.shape)
        vector = tf.concat([self.user_latent, self.item_latent], axis=-1)
        print(vector.shape)
        #vector = tf.multiply(self.user_latent, self.item_latent)
        #vectors = tf.matmul(user_latent, item_latent)
        #vector = tf.keras.layers.multiply([user_latent, item_latent])
        # MLP layers
        for idx in range(1,num_layer):
            W = tf.Variable(tf.random_uniform([int(vector.shape[1]),int(vector.shape[1])],-1.0,1.0), name='W_{}'.format(idx))
            b = tf.Variable(tf.constant(0.0, shape=[int(vector.shape[1])]), name='b')
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            vector = tf.nn.relu(tf.nn.xw_plus_b(vector, W, b, name='layer_{}'.format(idx)), name='relu')
            #layer = tf.nn.relu(layer, name='relu')

            #layer = tf.keras.layers.Dense(layer[idx], kernel_regularizer=l2(reg_layers[idx]), activation='relu')
            #vector = layer(vector)

        # Final prediction layer
        W = tf.Variable(tf.random_uniform([int(vector.shape[1]),1],-1.0,1.0), name='W_pred')
        b = tf.Variable(tf.constant(0.1, shape=[1],name='b'), name='b_pred')
        self.prediction = tf.nn.xw_plus_b(vector, W, b, name='pred_layer')
        #prediction = tf.keras.layers.Dense(1, activation='sigmoid', init='lecun_uniform', name='prediction')(vector)

        #model = tf.keras.models.Model(input=[user_input, item_imput], output=prediction) # 요거 텐서플로우선 지원 안함.

        #return model

def get_train_instances(train, num_negatives):
    user_input, item_input, labels = [], [], []
    num_users = train.shape[0]
    for (u,i) in train.keys():
        # positive instance
        user_input.append(u)
        item_input.append(i)
        labels.append(1)
        # negative_instance:
        for i in range(num_negatives):
            j = np.random.randint(num_items)
            while train.has_key((u,j)):
                j = np.random.randint(num_items)
            user_input.append(u)
            item_input.append(j)
            labels.append(0)
    return user_input, item_input, labels

if __name__=='__main__':
    args = parse_args()
    path = args.path
    dataset = args.dataset
    layers = eval(args.layers)
    reg_layers = eval(args.reg_layers)
    num_negatives = args.num_neg
    learner = args.learner
    learning_rate = args.lr
    batch_size = args.batch_size
    epochs = args.epochs
    verbose = args.verbose

    topK = 10
    evaluation_threads = 1 # mp.cpu_count()
    print("MLP arguments: %s " %(args))
    #model_out_file = ~ # Tune for Tensorflow

    # Loading data
    t1 = time()
    with tf.device('/gpu:0'):
        dataset = Dataset(args.path + args.dataset)
        train, testRatings, testNegatives = dataset.trainMatrix, dataset.testRatings, dataset.testNegatives
    num_users, num_items = train.shape

    print("Load data done [%.1f s]. #user=%d, #item=%d, #train=%d, #test=%d"
            %(time()-t1, num_users, num_items, train.nnz, len(testRatings)))

    np.random.seed(10)
    
    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=args.allow_soft_placement,
            log_device_placement=args.log_device_placement)

        sess = tf.Session(config=session_conf)
        
        with sess.as_default():
            model = MLP(num_users,num_items, layers, reg_layers)
            # Define Training Procedure
        
    '''
    # Build model
    model = get_model(num_users, num_items, layers, reg_layers)
    if learner.lower() == 'adagrad':
        model.compile(optimizer=Adagrad(~)) # Tune for Tensorflow
    elif learner.lower() == 'rmsprop':
        model.compile(optimizer=RMSProp(~)) # Tune for Tensorflow
    elif learner.lower() == 'adam':
        model.compile(optimizer=Adam(~)) # Tune for Tensorflow
    else:
        model.compie(optimizer=SGD~) # Tune for Tensorflow

    # Check Init performance
    t1 = time()
    (hits, ndcgs) = evaluate_model(model, testRatings, testNegatives, topK, evaluation_threads)
    hr, ndcg = np.array(hits).mean, np.array(ndcgs.mean())
    print("Init: HR = %.4f, NDCG = %.4f [$.1f]" % (hr, ndcg, time()-t1))

    # Train model
    best_hr, best_ndcg, best_iter = hr, ndcg, -1
    for epoch in range(epochs):
        t1 = time()
        # Generate training instances
        user_input, item_input, labels = get_train_instance(train, num_negatives)

        # Training
        hist = model.fit(~~) # Tune for Tensorflow
        t2 = time()

        if epoch % verbose == 0:
            (hits, ndcgs) = evaluate_model(model, testRatings, testNegatives, topK, evaluation_threads)
            hr, ndcg, loss = np.array(hits).mean(), np.array(ndcgs).mean(). hist.history['loss'][0]
            # wait....... for end tuning to tensorflow
    '''
