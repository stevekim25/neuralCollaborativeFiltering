import numpy as np

import tensorflow as tf
#from evaluate import evaluate_model
from Dataset import Dataset
from time import time
import datetime
import sys
import os
import GMF, MLP
from batchGen import batch_iter_per_epoch
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Run NeuMF.")
    parser.add_argument('--path', nargs='?', default='Data/',
                        help='Input data path.')
    parser.add_argument('--dataset', nargs='?', default='ml-1m',
                        help='Choose a dataset.')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs.')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size.')
    parser.add_argument('--num_factors', type=int, default=8,
                        help='Embedding size of MF model.')
    parser.add_argument('--layers', nargs='?', default='[64,32,16,8]',
                        help="MLP layers. Note that the first layer is the concatenation of user and item embeddings. So layers[0]/2 is the embedding size.")
    parser.add_argument('--reg_mf', type=float, default=0,
                        help='Regularization for MF embeddings.')                    
    parser.add_argument('--reg_layers', nargs='?', default='[0,0,0,0]',
                        help="Regularization for each MLP layer. reg_layers[0] is the regularization for embeddings.")
    parser.add_argument('--num_neg', type=int, default=4,
                        help='Number of negative instances to pair with a positive instance.')
    parser.add_argument('--l2_reg_lambda', type=float, default=.0,
                        help='L2 regularization lambda (default: 0.0)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate.')
    parser.add_argument('--learner', nargs='?', default='adam',
                        help='Specify an optimizer: adagrad, adam, rmsprop, sgd')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Show performance per X iterations')
    parser.add_argument('--out', type=int, default=1,
                        help='Whether to save the trained model.')
    parser.add_argument('--mf_pretrain', nargs='?', default='',
                        help='Specify the pretrain model file for MF part. If empty, no pretrain will be used')
    parser.add_argument('--mlp_pretrain', nargs='?', default='',
                        help='Specify the pretrain model file for MLP part. If empty, no pretrain will be used')
    parser.add_argument('--allow_soft_placement', type=bool, default=True,
                        help='Allow device soft device placement')
    parser.add_argument('--log_device_placement', type=bool, default=False,
                        help='Log placement of ops on devices')

    parser.add_argument("--display_every", type=int, default=1, help="Number of iterations to display training info.")
    #parser.add_argument("--evaluate_every", type=int, default=100, help="Evaluate model on dev set after this many steps (default: 100)")
    parser.add_argument("--checkpoint_every", type=int, default=1, help="Save model after this many steps (default: 100)")
    parser.add_argument("--num_checkpoints", type=int, default=5, help="Number of checkpoints to store (default: 5)")

    return parser.parse_args()

def init_normal(shape, name=None):
    return "glorot_uniform"

class NeuMF:
    def __init__(self, num_users, num_items,
                mf_dim=10, layers=[10], reg_layers=[0], reg_mf=0, l2_reg_lambda=0.0):
        assert len(layers) == len(reg_layers)
        num_layer = len(layers)

        l2_loss = tf.constant(0.0)
        # Input variables
        self.user_input = tf.placeholder(tf.int32, shape=[None,1], name='user_input') # check shape
        self.item_input = tf.placeholder(tf.int32, shape=[None,1], name='item_input') # check shape
        self.labels = tf.placeholder(tf.float32, shape=[None,1], name='labels') # check shape

        # Embedding layer
        with tf.device('/gpu:0'), tf.name_scope('mlp_user_embedding'):
            self.W_user = tf.Variable(tf.random_uniform([num_users, int(layers[0]/2)], -1.0, 1.0), name='W_mlp_user')
            MLP_Embedding_User = tf.nn.embedding_lookup(self.W_user, self.user_input)
        with tf.device('/gpu:0'), tf.name_scope('mlp_item_embedding'):
            self.W_item = tf.Variable(tf.random_uniform([num_items, int(layers[0]/2)], -1.0, 1.0), name='W_mlp_item')
            MLP_Embedding_Item = tf.nn.embedding_lookup(self.W_item, self.item_input)
        with tf.device('/gpu:0'), tf.name_scope('user_embedding'):
            self.W_user = tf.Variable(tf.random_uniform([num_users, mf_dim],-1.0,1.0), name='W_mf_user')
            MF_Embedding_User = tf.nn.embedding_lookup(self.W_user, self.user_input)
        with tf.device('/gpu:0'), tf.name_scope('item_embedding'):
            self.W_item = tf.Variable(tf.random_uniform([num_items, mf_dim],-1.0,1.0), name='W_mf_item')
            MF_Embedding_Item = tf.nn.embedding_lookup(self.W_item, self.item_input)

        # MF part
        self.mf_user_latent = tf.layers.Flatten()(MF_Embedding_User)
        self.mf_item_latent = tf.layers.Flatten()(MF_Embedding_Item)
        mf_vector = tf.multiply(self.mf_user_latent, self.mf_item_latent) # Element-wise product of user and item embeddings

        # MLP part
        self.mlp_user_latent = tf.layers.Flatten()(MLP_Embedding_User)
        self.mlp_item_latent = tf.layers.Flatten()(MLP_Embedding_Item)

        mlp_vector = tf.concat([self.mlp_user_latent, self.mlp_item_latent], axis=-1)
        # MLP layers
        for idx in range(1,num_layer):
            W = tf.Variable(tf.random_uniform([int(mlp_vector.shape[1]),int(mlp_vector.shape[1])],-0.25,0.25), name='W_mlp_{}'.format(idx))
            b = tf.Variable(tf.constant(0.1, shape=[int(mlp_vector.shape[1])]), name='b_mlp')
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            mlp_vector = tf.nn.relu(tf.nn.xw_plus_b(mlp_vector, W, b, name='layer_{}'.format(idx)), name='mlp_{}'.format(idx))

        predict_vector = tf.concat([mf_vector, mlp_vector], axis=-1)

        with tf.name_scope('prediction'):
            W = tf.Variable(tf.random_uniform([int(predict_vector.shape[-1]),1],-0.25,0.25), name='W_pred')
            b = tf.Variable(tf.constant(0.1, shape=[1]), name='b_pred')
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            #self.logits = tf.nn.xw_plus_b(predict_vector, W, b, name='logits')
            self.logits = tf.nn.xw_plus_b(predict_vector, W, b, name='logits')
            #self.predictions = tf.argmax(self.logits, axis=1, name='predictions')

        # Calculate mean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits, labels=self.labels)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        with tf.name_scope("accuracy"):
            self.accuracy = tf.reduce_mean(tf.square(self.logits - self.labels))
            #correct_predictions = tf.equal(self.predictions, self.labels)
            #self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32), name='accuracy')

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
            while (u, j) in train:
                #while train.has_key((u,j)):
                j = np.random.randint(num_items)
            user_input.append(u)
            item_input.append(j)
            labels.append(0)
    return user_input, item_input, labels

if __name__=='__main__':
    args = parse_args()
    num_epochs = args.epochs
    batch_size = args.batch_size
    mf_dim = args.num_factors
    layers = eval(args.layers)
    reg_mf = args.reg_mf
    reg_layers = eval(args.reg_layers)
    num_negatives= args.num_neg
    learning_rate = args.lr
    learner = args.learner
    verbose = args.verbose
    mf_pretrain = args.mf_pretrain
    mlp_pretrain = args.mlp_pretrain

    topK = 10
    evaluation_threads = 1
    print("NeuMF argument: %s " % (args))
    
    t1 = time()
    dataset = Dataset(args.path + args.dataset)
    train, testRatings, testNegatives = dataset.trainMatrix, dataset.testRatings, dataset.testNegatives
    num_users, num_items = train.shape
    print("Load data done [%.1f s]. #user=%d, #item=%d, #train=%d, #test=%d"
            % (time()-t1, num_users, num_items, train.nnz, len(testRatings)))

    # Build model
    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=args.allow_soft_placement,
            log_device_placement=args.log_device_placement)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            neuMF = NeuMF(num_users, num_items, mf_dim, layers, reg_layers, reg_mf, args.l2_reg_lambda)

            # Define Training Procedure
            global_step = tf.Variable(0, name='global_step', trainable=False)

            if learner.lower() == 'adagrad':
                train_op = tf.train.AdagradOptimizer(learning_rate=learning_rate).minimize(neuMF.loss, global_step=global_step)
            elif learner.lower() == 'rmsprop':
                train_op = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(neuMF.loss, global_step=global_step)
            elif learner.lower() == 'adam':
                train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(neuMF.loss, global_step=global_step)
            else:
                train_op = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(neuMF.loss, global_step=global_step)

            # Output directory for models and summaries
            timestamp = str(int(time()))
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
            print("Writing to {}\n".format(out_dir))

            # Summaries for loss and accuracy
            loss_summary = tf.summary.scalar("loss", neuMF.loss)
            acc_summary = tf.summary.scalar("accuracy", neuMF.accuracy)

            # Train Summaries
            train_summary_op = tf.summary.merge([loss_summary, acc_summary])
            train_summary_dir = os.path.join(out_dir, "summaries", "train")
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)
            
            # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=args.num_checkpoints)

            # Initialize all variables
            sess.run(tf.global_variables_initializer())

            user_input, item_input, labels = get_train_instances(train, num_negatives)

            #batches = batch_iter(
            #    list(zip(user_input, item_input, labels)), batch_size, num_epochs)

            for epoch in range(num_epochs):
                batches = batch_iter_per_epoch(
                    list(zip(user_input, item_input, labels)), batch_size)
                losses = 0.0
                for batch in batches:
                    user_batch, item_batch, label_batch = zip(*batch)
                    user_batch = np.array(user_batch)
                    user_batch = user_batch.reshape(len(user_batch),1)
                    item_batch = np.array(item_batch)
                    item_batch = user_batch.reshape(len(item_batch),1)
                    label_batch = np.array(label_batch)
                    label_batch = label_batch.reshape(len(label_batch),1)

                    feed_dict = {
                        neuMF.user_input: user_batch,
                        neuMF.item_input: item_batch,
                        neuMF.labels: label_batch
                    }
                    _, step, summaries, loss, accuracy = sess.run(
                        [train_op, global_step, train_summary_op, neuMF.loss, neuMF.accuracy], feed_dict)
                    train_summary_writer.add_summary(summaries, step)
                    #print(predictions)
                    losses += loss

                    # Training log display
                if epoch % args.display_every == 0:
                    time_str = datetime.datetime.now().isoformat()
                    print("{}: step {}, loss {:g}, acc {:g}".format(time_str, epoch, losses, accuracy))

                if epoch % args.checkpoint_every == 0:
                    path = saver.save(sess, checkpoint_prefix, global_step=step)
                    print("Saved model checkpoint to {}\n".format(path))
