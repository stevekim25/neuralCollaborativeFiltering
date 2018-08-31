import tensorflow as tf
import numpy as np
import os
import argparse
import time
import datetime
from Dataset import Dataset
from NeuMF import NeuMF
from tensorflow.contrib import learn

def parse_args():
    parser = argparse.ArgumentParser(description="Eval NeuMF.")
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
    parser.add_argument('--checkpoint_dir', type=str, default='',
                        help='Checkpoint directory from training run')
    parser.add_argument('--allow_soft_placement', type=bool, default=True,
                        help='Allow device soft device placement')
    parser.add_argument('--log_device_placement', type=bool, default=False,
                        help='Log placement of ops on devices')

    parser.add_argument("--display_every", type=int, default=1, help="Number of iterations to display training info.")
    #parser.add_argument("--evaluate_every", type=int, default=100, help="Evaluate model on dev set after this many steps (default: 100)")
    parser.add_argument("--checkpoint_every", type=int, default=1, help="Save model after this many steps (default: 100)")
    parser.add_argument("--num_checkpoints", type=int, default=5, help="Number of checkpoints to store (default: 5)")

    return parser.parse_args()

def eval():
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
    testRatings, testNegatives = dataset.testRatings, dataset.testNegatives
    
    print("Load data done [%.1f s]. #user=%d, #item=%d, #test=%d"
            % (time()-t1, num_users, num_items, len(testRatings)))

    for idx in range(len(testRatings)):
        rating = testRatings[idx]
        items = testNegatives[idx]
        u = rating[0]
        gtItem = rating[1]
        items.append(gtItem)

        users_inp = np.full(len(items),u,dtype='int32')
        items_inp = np.array(items)

    checkpoint_file = tf.train.latest_checkpoint(args.checkpoint_dir)
    graph = tf.Graph()
    with graph.as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=args.allow_soft_placement,
            log_device_placement=args.log_device_placement)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            saver = tf.train.import_meta_graph("{}/meta".format(checkpoint_file))
            saver.restore(sess, checkpoint_file)

            # Get the placeholders from the graph by name
            user_input = graph.get_operation_by_name('user_input').outputs[0]
            item_input = graph.get_operation_by_name('item_input').outputs[0]

            predictions = graph.get_operation_by_name('output/predictions').outputs[0]

            all_predictions = sess.run(predictions, {user_input: ~})

# 여기서부터 평가 시작하면됨.


 
if __name__=='__main__':
    eval()
