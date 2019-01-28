import tensorflow as tf
import cv2
from pixel2mesh.models import GCN
from pixel2mesh.fetcher import *
from pixel2mesh.cd_dist import *
import os
import datetime as dt
import pandas as pd
import numpy as np

import cPickle as pickle 
import re

# Use data downloaded from AtlasNet paper
path_data = "../../../papier_mache/AtlasNet/data/"

def get_time():
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M") 

cat = {}
with open(os.path.join(path_data, 'synsetoffset2category.txt'), 'r') as f:
    for line in f:
        ls = line.strip().split()
        cat[ls[0]] = ls[1]

# Set random seed
seed = 1024
np.random.seed(seed)
tf.set_random_seed(seed)

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('object_random', 0, '0 for not random, a positive integer for number of random objects')
flags.DEFINE_string('object_cat', 'plane', 'name of category')
flags.DEFINE_string('object_id', '1a888c2c86248bbcf2b0736dd4d8afe0', 'id of the set of images')
flags.DEFINE_string('object_img', None, 'name of image .dat')
flags.DEFINE_string('output_path', '../../../comparison/output_pixel_2_mesh', 'name of image')
flags.DEFINE_string('model', 'gcn', 'name of the model')
flags.DEFINE_float('learning_rate', 0., 'Initial learning rate.')
flags.DEFINE_integer('hidden', 192, 'Number of units in  hidden layer.')
flags.DEFINE_integer('feat_dim', 963, 'Number of units in perceptual feature layer.')
flags.DEFINE_integer('coord_dim', 3, 'Number of units in output layer.') 
flags.DEFINE_float('weight_decay', 5e-6, 'Weight decay for L2 loss.')
flags.DEFINE_integer('chamfer_points', 10000, 'number of points used to compute chamfer distance')

fscore_thresh = [1e-4, 2e-4]

if not FLAGS.object_random:
    # Check that the desired object does exist
    if not FLAGS.object_cat in cat.keys():
        raise ValueError("Please specify a category among %s" % cat.keys())
    path_cat = os.path.join(path_data, "ShapeNet/ShapeNetRendering", cat[FLAGS.object_cat])
    path_id = os.path.join(path_cat, FLAGS.object_id, "rendering")
    if not FLAGS.object_id in os.listdir(path_cat):
        raise ValueError("Please specify an object id that is in the category %s" % cat[FLAGS.object_cat])
    
    if FLAGS.object_img is not None and not FLAGS.object_img in os.listdir(path_id):
        raise ValueError("Please specify an object image that is in the id %s of category %s" % (FLAGS.object_id, cat[FLAGS.object_cat]))
    print("Manually selected object %s of category %s (%s)" % (FLAGS.object_id, FLAGS.object_cat,
                                                               cat[FLAGS.object_cat]))

model_split = FLAGS.model.split("_")
train_cat = [x for x in model_split if x in cat.keys()]

if len(train_cat)==0:
    train_cat = None
    file_list = "utils/train_list.txt"
    training = "all_cat"
else:
    file_list = "utils/train_list_%s.txt" % "_".join(train_cat)
    training = "_".join(train_cat)

re1='.*?'	# Non-greedy match on filler
re2='\\d+'	# Uninteresting: int
re3='.*?'	# Non-greedy match on filler
re4='(\\d+)'	# Integer Number 1
re5='((?:[a-z][a-z]*[0-9]+[a-z0-9]*))'	# Alphanum 1

rg = re.compile(re1+re2+re3+re4+re5,re.IGNORECASE|re.DOTALL)

list_train_object_id = []
with open(file_list, 'r') as f:
    while(True):
        line = f.readline().strip()
        if not line:
            break
        else:
            rs = rg.search(line)
            object_id = str(rs.group(1)) + rs.group(2)
            list_train_object_id.append(object_id)

# Define placeholders(dict) and model
num_blocks = 3
num_supports = 2
placeholders = {
    'features': tf.placeholder(tf.float32, shape=(None, 3)), # initial 3D coordinates
    'img_inp': tf.placeholder(tf.float32, shape=(224, 224, 3)), # input image to network
    'labels': tf.placeholder(tf.float32, shape=(None, 6)), # ground truth (point cloud with vertex normal)
    'support1': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)], # graph structure in the first block
    'support2': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)], # graph structure in the second block
    'support3': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)], # graph structure in the third block
    'faces': [tf.placeholder(tf.int32, shape=(None, 4)) for _ in range(num_blocks)], # helper for face loss (not used)
    'edges': [tf.placeholder(tf.int32, shape=(None, 2)) for _ in range(num_blocks)], # helper for normal loss
    'lape_idx': [tf.placeholder(tf.int32, shape=(None, 10)) for _ in range(num_blocks)], # helper for laplacian regularization
    'pool_idx': [tf.placeholder(tf.int32, shape=(None, 2)) for _ in range(num_blocks-1)] # helper for graph unpooling
}
model = GCN(placeholders, logging=True, name=FLAGS.model)

# Construct feed dictionary
def construct_feed_dict(pkl, placeholders):
	coord = pkl[0]
	pool_idx = pkl[4]
	faces = pkl[5]
	lape_idx = pkl[7]
	edges = []
	for i in range(1,4):
		adj = pkl[i][1]
		edges.append(adj[0])
	feed_dict = dict()
	feed_dict.update({placeholders['features']: coord})
	feed_dict.update({placeholders['edges'][i]: edges[i] for i in range(len(edges))})
	feed_dict.update({placeholders['faces'][i]: faces[i] for i in range(len(faces))})
	feed_dict.update({placeholders['pool_idx'][i]: pool_idx[i] for i in range(len(pool_idx))})
	feed_dict.update({placeholders['lape_idx'][i]: lape_idx[i] for i in range(len(lape_idx))})
	feed_dict.update({placeholders['support1'][i]: pkl[1][i] for i in range(len(pkl[1]))})
	feed_dict.update({placeholders['support2'][i]: pkl[2][i] for i in range(len(pkl[2]))})
	feed_dict.update({placeholders['support3'][i]: pkl[3][i] for i in range(len(pkl[3]))})
	return feed_dict

def load_image(img_path):
	img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
	if img.shape[2] == 4:
		img[np.where(img[:,:,3]==0)] = 255
	img = cv2.resize(img, (224,224))
	img = img.astype('float32')/255.0
	return img[:,:,:3]

# Load data, initialize session
config=tf.ConfigProto()
config.gpu_options.allow_growth=True
config.allow_soft_placement=True
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())
model.load(sess)

# xyz1:dataset_points * 3, xyz2:query_points * 3
xyz1=tf.placeholder(tf.float32,shape=(None, 3))
xyz2=tf.placeholder(tf.float32,shape=(None, 3))
# chamfer distance
dist1,idx1,dist2,idx2 = nn_distance(xyz1, xyz2)

pkl = pickle.load(open('utils/ellipsoid/info_ellipsoid.dat', 'rb'))
feed_dict = construct_feed_dict(pkl, placeholders)

##########################
# Prepare functions and 
# files to compute metrics 
# on outputs
##########################

def load_image_point_cloud(path_pkl):
    pkl = pickle.load(open(path_pkl, 'rb'))
    img = pkl[0].astype('float32')/255.0
    label = pkl[1]
    return img, label

def center_max_scale(arr):
    arr = arr - arr.mean(axis=0)
    arr = arr/arr.max(axis=0)
    return arr

def f_score(label, predict, dist_label, dist_pred, threshold):
    num_label = label.shape[0]
    num_predict = predict.shape[0]

    f_scores = []
    for i in range(len(threshold)):
        num = len(np.where(dist_label <= threshold[i])[0])
        recall = 100.0 * num / num_label
        num = len(np.where(dist_pred <= threshold[i])[0])
        precision = 100.0 * num / num_predict

        f_scores.append((2*precision*recall)/(precision+recall+1e-8))
    return np.array(f_scores)

def apply_model(df_results):
    # Apply the model
    #   - 1 image if specified in input
    #   - the set of images of 1 object if only object id specified

    print("="*40)
    if FLAGS.object_id in list_train_object_id:
        print("%s is train" % FLAGS.object_id)
        origin = 'train'
    else:
        print("%s is test" % FLAGS.object_id)
        origin = 'test'
    print("="*40)

    if FLAGS.object_img is None:
        fns_img = ["0%d.dat" % i for i in range(10)] + ["%d.dat" % i for i in range(10, 24)] 
        
        for fn in fns_img:
            img_inp, vert_truth = load_image_point_cloud(path_pkl + "_" + fn)
            feed_dict.update({placeholders['img_inp']: img_inp})
            
            #forward pass
            vert_pred = sess.run(model.output3, feed_dict=feed_dict)
            vert_pred = np.hstack((np.full([vert_pred.shape[0],1], 'v'), vert_pred))
            face = np.loadtxt('utils/ellipsoid/face3.obj', dtype='|S32')
            mesh = np.vstack((vert_pred, face))
    
            # Save output 3D model
            output_fn = os.path.join(output_dir, FLAGS.model + "_" + fn.replace('.dat', '.obj'))
            np.savetxt(output_fn, mesh, fmt='%s', delimiter=' ')

            # Random picks min(FLAGS.chamfer_points, n_vert_truth) points
            vert_truth = vert_truth[:, :3]
            n_vert_truth = vert_truth.shape[0]
            rand_rows = np.random.permutation(n_vert_truth)
            vert_truth = vert_truth[rand_rows[:min(FLAGS.chamfer_points, n_vert_truth)], :3]
            print("Number of vertices in label point cloud %d" % n_vert_truth)

            # Random picks min(FLAGS.chamfer_points, n_vert_pred) points
            vert_pred = vert_pred[:, 1:].astype(np.float32)
            n_vert_pred = vert_pred.shape[0]
            print("Number of vertices in predicted point cloud %d" % n_vert_pred)
            rand_rows = np.random.permutation(n_vert_pred)
            vert_pred = vert_pred[rand_rows[:min(FLAGS.chamfer_points, n_vert_pred)], :3]
            
            # Compute Chamfer distance
            dist1, idx1, dist2, idx2 = nn_distance(vert_truth, vert_pred)
            d1,i1,d2,i2 = sess.run([dist1,idx1,dist2,idx2], feed_dict={xyz1:vert_truth,xyz2:vert_pred})
            dist = np.mean(d1) + np.mean(d2)

            # Compute f_score
            fscores = f_score(vert_truth, vert_pred, d1, d2, fscore_thresh)
    
            row_results = {'training': training, 
                           'paper': 'pixel_2_mesh', 
                           'model': FLAGS.model, 
                           'cat': FLAGS.object_cat, 
                           'id': FLAGS.object_id, 
                           'img': fn, 
                           'origin': origin,
                           'chamfer_points': FLAGS.chamfer_points, 
                           'chamfer(x1000)': dist*1000, 
                           'output': FLAGS.model + "_" + fn.replace('.dat', '.obj'), 
                           'date': get_time()}
            
            for th, f in zip(fscore_thresh, fscores):
                row_results['fscore(%.1g)' % th] = f
                print("fscore (%.1g) between ground truth and %s/%s is: %.3g" % (th, FLAGS.object_id, fn, f))
                
            df_results = df_results.append(row_results, ignore_index=True)
            print("Chamfer distance (x1000) between ground truth and %s/%s is: %.3g" % (FLAGS.object_id, fn, dist*1000))
    
        subset = [x for x in df_results.columns if x not in ["chamfer(x1000)", "date"] and "fscore" not in x]           
        df_results.drop_duplicates(subset=subset, keep="first", inplace=True)
        df_results.to_csv("../../../comparison/df_results.csv", header=True, index=False)
        print("Done transforming %s ! Check out results in comparison/output_pixel_2_mesh/" % FLAGS.object_id)
    else:
        img_inp, vert_truth = load_image_point_cloud(path_pkl + "_" + FLAGS.object_img)
        feed_dict.update({placeholders['img_inp']: img_inp})
        
        #forward pass
        vert_pred = sess.run(model.output3, feed_dict=feed_dict)
        vert_pred = np.hstack((np.full([vert_pred.shape[0],1], 'v'), vert_pred))
        face = np.loadtxt('utils/ellipsoid/face3.obj', dtype='|S32')
        mesh = np.vstack((vert_pred, face))
        
        # Save output 3D model
        output_fn = os.path.join(output_dir, FLAGS.model + FLAGS.object_img.replace('.dat', '.obj'))
        np.savetxt(output_fn, mesh, fmt='%s', delimiter=' ')

        # Random picks min(FLAGS.chamfer_points, n_vert_truth) points
        vert_truth = vert_truth[:, :3]
        n_vert_truth = vert_truth.shape[0]
        rand_rows = np.random.permutation(n_vert_truth)
        vert_truth = vert_truth[rand_rows[:min(FLAGS.chamfer_points, n_vert_truth)], :3]
        print("Number of vertices in label point cloud %d" % n_vert_truth)
                
        # Random picks min(FLAGS.chamfer_points, n_vert_pred) points
        n_vert_pred = vert_pred.shape[0]
        print("Number of vertices in predicted point cloud %d" % n_vert_pred)
        rand_rows = np.random.permutation(n_vert_pred)
        vert_pred = vert_pred[rand_rows[:min(FLAGS.chamfer_points, n_vert_pred)], :3]
            
        # Compute Chamfer distance
        dist1, idx1, dist2, idx2 = nn_distance(vert_truth, vert_pred)
        d1,i1,d2,i2 = sess.run([dist1,idx1,dist2,idx2], feed_dict={xyz1:vert_truth,xyz2:vert_pred})
        dist = np.mean(d1) + np.mean(d2)

        # Compute f_score
        fscores = f_score(vert_truth, vert_pred, d1, d2, fscore_thresh)

        print("Chamfer distance (x1000) between ground truth and %s/%s is: %.3g" % (FLAGS.object_id, FLAGS.object_img, dist*1000))
        
        row_results = {'training': training, 
                       'paper': 'pixel_2_mesh', 
                       'model': FLAGS.model, 
                       'cat': FLAGS.object_cat, 
                       'id': FLAGS.object_id, 
                       'img': FLAGS.object_img, 
                       'origin': origin,
                       'chamfer_points': FLAGS.chamfer_points, 
                       'chamfer(x1000)': dist*1000, 
                       'output':  FLAGS.model + "_" + FLAGS.object_img.replace('.dat', '.obj'),  
                       'date': get_time()}
        
        for th, f in zip(fscore_thresh, fscores):
            row_results['fscore(%.1g)' % th] = f
            print("fscore (%.1g) between ground truth and %s/%s is: %.3g" % (th, FLAGS.object_id, FLAGS.object_img, f))

        df_results = df_results.append(row_results, ignore_index=True)
        print("Chamfer distance (x1000) between ground truth and %s/%s is: %.3g" % (FLAGS.object_id, FLAGS.object_img, dist*1000))
        
        subset = [x for x in df_results.columns if x not in ["chamfer(x1000)", "date"] and "fscore" not in x]           
        df_results.drop_duplicates(subset=subset, keep="first", inplace=True)
        df_results.to_csv("../../../comparison/df_results.csv", header=True, index=False)
        print("Done transforming %s ! Check out results in comparison/output_pixel_2_mesh/" % FLAGS.object_id)

def object_select_random():
    path_ply = ""
    while not os.path.exists(path_ply):
        object_cat = np.random.choice(list(cat.keys()))
        path_cat =  os.path.join(path_data, "ShapeNet/ShapeNetRendering", cat[object_cat])
        object_id = np.random.choice(os.listdir(path_cat))
        path_pkl = "data/ShapeNetTrain/%s_%s" % (cat[object_cat], object_id)
    print("Randomly selected object %s of category %s (%s)" % (object_id, object_cat, cat[object_cat]))
    return object_cat, path_cat, object_id, path_pkl

#################
# Apply the model
#################

# Load or create csv file used to save results
if not os.path.exists("../../../comparison/df_results.csv"):
    df_results = pd.DataFrame(columns=['training', 'paper', 'model', 'cat', 'id', 'img', 'chamfer_points',
                                       'chamfer(x1000)', 'output', 'date'])
else:
    df_results = pd.read_csv("../../../comparison/df_results.csv", header='infer')

if FLAGS.object_random == -1:
    # Execute the selected models on all objects found in df_results.csv
    all_ids = set(df_results.id.unique())
    model_ids = set(df_results.loc[df_results.model==FLAGS.model, "id"].unique())

    for object_id in set(all_ids).difference(set(model_ids)):
        FLAGS.object_id = object_id
        FLAGS.object_cat = df_results.loc[df_results.id == object_id].cat.values[0]
        
        path_pkl = "data/ShapeNetTrain/%s_%s" % (cat[FLAGS.object_cat], FLAGS.object_id)
    
        # Create output directory if it does not exist already
        output_dir = os.path.join(FLAGS.output_path, cat[FLAGS.object_cat])
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
    
        output_dir = os.path.join(output_dir, FLAGS.object_id)
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
    
        apply_model(df_results)
        df_results = pd.read_csv("../../../comparison/df_results.csv", header='infer')


elif not FLAGS.object_random:
    # Path to object
    path_pkl = "data/ShapeNetTrain/%s_%s" % (cat[FLAGS.object_cat], FLAGS.object_id)

    # Create output directory if it does not exist already
    output_dir = os.path.join(FLAGS.output_path, cat[FLAGS.object_cat])
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    
    output_dir = os.path.join(output_dir, FLAGS.object_id)
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    apply_model(df_results)
    df_results = pd.read_csv("../../../comparison/df_results.csv", header='infer')
else:
    while FLAGS.object_random > 0:
        # Randomly select an object category and object id
        FLAGS.object_cat, path_cat, FLAGS.object_id, path_pkl =  object_select_random()
        
        # Create output directory if it does not exist already
        output_dir = os.path.join(FLAGS.output_path, cat[FLAGS.object_cat])
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        
        output_dir = os.path.join(output_dir, FLAGS.object_id)
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
    
        apply_model(df_results)
        df_results = pd.read_csv("../../../comparison/df_results.csv", header='infer')
        FLAGS.object_random = FLAGS.object_random - 1 
