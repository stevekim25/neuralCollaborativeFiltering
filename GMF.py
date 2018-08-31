import numpy as np
import tensorflow as tf
from Dataset import Dataset
#from evaluate import evaluate_model
from time import time
import multiprocessing as mp
import sys
import math
import argparse

#################### Arguments ####################
def parse_args():
    parser = argparse.ArgumentParser(description="Run GMF.")
    parser.add_argument('--path', nargs='?', default='Data/',
                        help='Input data path.')
    parser.add_argument('--dataset', nargs='?', default='ml-1m',
                        help='Choose a dataset.')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs.')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size.')
    parser.add_argument('--num_factors', type=int, default=8,
                        help='Embedding size.')
    parser.add_argument('--regs', nargs='?', default='[0,0]',
                        help="Regularization for user and item embeddings.")
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

class GMF:
    def __init__(self, num_users, num_items, latent_dim, regs=[0,0]):
        # Input variables
        l2_loss = tf.constant(0.0)
        self.user_input = tf.placeholder(tf.int32, shape=[None,1], name='user_input')
        self.item_input = tf.placeholder(tf.int32, shape=[None,1], name='item_input')

        with tf.device('/gpu:0'), tf.name_scope('user_embedding'):
            self.W_user = tf.Variable(tf.random_uniform([num_users, latent_dim],-1.0,1.0), name='W_user')
            MF_Embedding_User = tf.nn.embedding_lookup(self.W_user, self.user_input)
        with tf.device('/gpu:0'), tf.name_scope('item_embedding'):
            self.W_item = tf.Variable(tf.random_uniform([num_items, latent_dim],-1.0,1.0), name='W_item')
            MF_Embedding_Item = tf.nn.embedding_lookup(self.W_item, self.item_input)

        # Crucial to flatten an embedding vector!
        self.user_latent = tf.layers.Flatten()(MF_Embedding_User)
        self.item_latent = tf.layers.Flatten()(MF_Embedding_Item)
        
        # Element-wise product of user and item embeddings 
        predict_vector = tf.multiply(self.user_latent, self.item_latent)
        print(predict_vector.shape)
        # Final prediction layer
        with tf.name_scope("prediction"):
            W = tf.Variable(tf.random_uniform([int(predict_vector.shape[-1]),1],-0.25,0.25), name='W_pred')
            b = tf.Variable(tf.constant(0.0, shape=[1]),name='b')
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            
            self.prediction = tf.sigmoid(tf.nn.xw_plus_b(predict_vector, W, b, name='pred_layer'))

        # Plus: loss and accuracy
        #model = Model(input=[user_input, item_input],
        #              output=prediction)
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
    num_factors = args.num_factors
    regs = eval(args.regs)
    num_negatives = args.num_neg
    learner = args.learner
    learning_rate = args.lr
    batch_size = args.batch_size
    epochs = args.epochs
    verbose = args.verbose

    topK = 10
    evaluation_threads = 1 # mp.cpu_count()
    print("GMF arguments: %s " %(args))
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
            model = GMF(num_users,num_items, num_factors, regs)

