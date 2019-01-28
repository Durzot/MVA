from __future__ import print_function
import sys
import argparse
import random
import numpy as np
import datetime as dt
from torch.autograd import Variable
import sys
import time
sys.path.append('./auxiliary/')
from dataset import *
from model import *
from utils import *
from ply import *

from PIL import Image
import torchvision.transforms as transforms
import pandas as pd


#################################################################
# This script runs an AtlasNet model and compute Chamfer distance
# and fscore resulting on random or selected inputs
#################################################################

def get_time():
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M") 

cat = {}
with open(os.path.join('./data/synsetoffset2category.txt'), 'r') as f:
    for line in f:
        ls = line.strip().split()
        cat[ls[0]] = ls[1]

parser = argparse.ArgumentParser()
parser.add_argument('--object_random', type=int, default=0, help='0 for not random, a positive integer for number of random objects')
parser.add_argument('--object_cat', type=str, default="plane", help='name of category')
parser.add_argument('--object_id', type=str, default="1a888c2c86248bbcf2b0736dd4d8afe0", help='id of the set of images')
parser.add_argument('--object_img', type=str, default=None, help='name of the image .png')
parser.add_argument('--output_path', type=str, default="../../comparison/output_atlas_net", help='path to output models')
parser.add_argument('--model', type=str, default = 'trained_models/svr_atlas_25.pth',  help='your path to the trained model')
parser.add_argument('--num_points', type=int, default = 2500,  help='number of points fed to poitnet')
parser.add_argument('--gen_points', type=int, default = 30000,  help='number of points to generate')
parser.add_argument('--nb_primitives', type=int, default = 25,  help='number of primitives')
parser.add_argument('--chamfer_points', type=int, default = 2500,  help='number of points used to compute chamfer distance')
parser.add_argument('--fscore_thresh', type=float, nargs='+', default=[1e-3, 2e-3], help='list thresholds for f_score')
parser.add_argument('--cuda', type=int, default = 0,  help='use cuda')
parser.add_argument('--accelerated_chamfer', type=int, default=0,  help='use custom build accelarated chamfer')

opt = parser.parse_args()

if opt.accelerated_chamfer and not opt.cuda:
    raise ValueError("Please activate cuda with --cuda 1 if you want to use cuda accelerated chamfer distance")

if not opt.object_random:
    # Check that the desired object does exist
    if not opt.object_cat in cat.keys():
        raise ValueError("Please specify a category among %s" % cat.keys())
    path_cat = os.path.join("./data/ShapeNet/ShapeNetRendering", cat[opt.object_cat])
    path_id = os.path.join(path_cat, opt.object_id, "rendering")
    if not opt.object_id in os.listdir(path_cat):
        raise ValueError("Please specify an object id that is in the category %s" % cat[opt.object_cat])
    if opt.object_img is not None and not opt.object_img in os.listdir(path_id):
        raise ValueError("Please specify an object image that is in the id %s of category %s" % (opt.object_id, cat[opt.object_cat]))
    print("Manually selected object %s of category %s (%s)" % (opt.object_id, opt.object_cat, cat[opt.object_cat]))


model_name_split = opt.model.split("/")[1][:-4].split("_")
train_cat = [x for x in model_name_split if x in cat.keys()]

if len(train_cat)==0:
    train_cat = None
    training = "all_cat"
else:
    training = "_".join(train_cat)

# Load data to know which one is train, which one is test
dataset_train = ShapeNet(SVR=True, normal = False, class_choice = train_cat, train=True)
dataset_test = ShapeNet(SVR=True, normal = False, class_choice=None, train=False)

list_train_object_id = []
for _,_,_,_,object_id in dataset_train.datapath:
    list_train_object_id.append(object_id)

list_test_object_id = []
for _,_,_,_,object_id in dataset_test.datapath:
    list_test_object_id.append(object_id)

blue = lambda x:'\033[94m' + x + '\033[0m'
opt.manualSeed = random.randint(1, 10000) # fix seed
np.random.seed(opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

network = SVR_AtlasNet(num_points = opt.num_points, nb_primitives = opt.nb_primitives, cuda = opt.cuda)
if opt.cuda:
    network.cuda()

network.apply(weights_init)
if opt.model != '':
    if opt.cuda:
        network.load_state_dict(torch.load(opt.model))
    else:
        network.load_state_dict(torch.load(opt.model, map_location='cpu'))
    print("previous weight loaded")
    
network.eval()
grain = int(np.sqrt(opt.gen_points/opt.nb_primitives))-1
grain = grain*1.0

#generate regular grid
faces = []
vertices = []
face_colors = []
vertex_colors = []
colors = get_colors(opt.nb_primitives)

for i in range(0,int(grain + 1 )):
        for j in range(0,int(grain + 1 )):
            vertices.append([i/grain,j/grain])

for prim in range(0, opt.nb_primitives):
    for i in range(0,int(grain + 1)):
        for j in range(0,int(grain + 1)):
            vertex_colors.append(colors[prim])
    for i in range(1,int(grain + 1)):
        for j in range(0,(int(grain + 1)-1)):
            faces.append([(grain+1)*(grain+1)*prim + j+(grain+1)*i, (grain+1)*(grain+1)*prim + j+(grain+1)*i + 1, (grain+1)*(grain+1)*prim + j+(grain+1)*(i-1)])
    for i in range(0,(int((grain+1))-1)):
        for j in range(1,int((grain+1))):
            faces.append([(grain+1)*(grain+1)*prim + j+(grain+1)*i, (grain+1)*(grain+1)*prim + j+(grain+1)*i - 1, (grain+1)*(grain+1)*prim + j+(grain+1)*(i+1)])

df_faces = np.zeros((len(faces), 4)) + 3
df_faces[:,1:] = np.array(faces)
df_faces = pd.DataFrame(df_faces.astype(int))

grid = [vertices for i in range(0,opt.nb_primitives)]
grid_pytorch = torch.Tensor(int(opt.nb_primitives*(grain+1)*(grain+1)),2)
for i in range(opt.nb_primitives):
    for j in range(int((grain+1)*(grain+1))):
        grid_pytorch[int(j + (grain+1)*(grain+1)*i),0] = vertices[j][0]
        grid_pytorch[int(j + (grain+1)*(grain+1)*i),1] = vertices[j][1]

##########################
# Prepare functions and 
# files to compute metrics 
# on outputs
##########################

my_transforms = transforms.Compose([
                 transforms.CenterCrop(127),
                 transforms.Resize(size =  224, interpolation = 2),
                 transforms.ToTensor(),
                 # normalize,
            ])

def get_point_cloud(path):
    with open(path, 'r') as file:
        lines = file.readlines()[12:] # Skip 12 first lines
        return np.loadtxt(lines).astype(np.float32)

def bdiag(x):
    # Return batch diagonal matrix
    bs, n, _ = x.size()
    xdiag = torch.zeros(bs, n).type(x.type())
    for b in range(bs):
        xdiag[b] = torch.diagonal(x[b, :, :])
    return xdiag

if opt.accelerated_chamfer:
    sys.path.append("./extension/")
    import dist_chamfer as ext
    distChamfer =  ext.chamferDist()

    def distChamferFscore(x, y, thresholds=[1e-3, 2e-3]):
        """ 
        Chamfer distance defined as
        1/|P| \sum_{p \in P} min_{q \in Q} ||p-q||^2 + 1/|Q| \sum_{q \in Q} min_{p \in P} ||p-q||^2 
        """
        bsx, nx, px = x.size()
        bsy, ny, py = y.size()
        if bsx != bsy:
            raise ValueError("Please send x and y with same batch dimension %d != %d" % (bsx, bsy))
        elif px != py:
            raise ValueError("Please use points from the same vector space %d != %d" %(px, py)) 

        dx, dy = distChamfer(x, y)

        fscores = []
        for b in range(bsx):
            b_fscores = []
            for i in range(len(thresholds)):
                num = len(np.where(dx.cpu().detach().numpy() <= thresholds[i])[0])
                recall = 100.0 * num / nx
                num = len(np.where(dy.cpu().detach().numpy() <= thresholds[i])[0])
                precision = 100.0 * num / ny
                b_fscores.append((2*precision*recall)/(precision+recall+1e-8))
            fscores.append(b_fscores)
    
        return dx, dy, fscores

else:
    def distChamfer(x, y):
        """ 
        Chamfer distance defined as
        1/|P| \sum_{p \in P} min_{q \in Q} ||p-q||^2 + 1/|Q| \sum_{q \in Q} min_{p \in P} ||p-q||^2 
        """
        bsx, nx, px = x.size()
        bsy, ny, py = y.size()
        if bsx != bsy:
            raise ValueError("Please send x and y with same batch dimension %d != %d" % (bsx, bsy))
        elif px != py:
            raise ValueError("Please use points from the same vector space %d != %d" %(px, py)) 
        xx = torch.bmm(x, x.transpose(2,1))
        yy = torch.bmm(y, y.transpose(2,1))
        zz = torch.bmm(x, y.transpose(2,1))
        rx = bdiag(xx).unsqueeze(1).expand(bsx, ny, nx)
        ry = bdiag(yy).unsqueeze(1).expand(bsx, nx, ny)
        P = (rx.transpose(2,1) + ry - 2*zz)
        return P.min(1)[0], P.min(2)[0]

    def distChamferFscore(x, y, thresholds=[1e-3, 2e-3]):
        """ 
        Chamfer distance defined as
        1/|P| \sum_{p \in P} min_{q \in Q} ||p-q||^2 + 1/|Q| \sum_{q \in Q} min_{p \in P} ||p-q||^2 
        """
        bsx, nx, px = x.size()
        bsy, ny, py = y.size()
        if bsx != bsy:
            raise ValueError("Please send x and y with same batch dimension %d != %d" % (bsx, bsy))
        elif px != py:
            raise ValueError("Please use points from the same vector space %d != %d" %(px, py)) 
        xx = torch.bmm(x, x.transpose(2,1))
        yy = torch.bmm(y, y.transpose(2,1))
        zz = torch.bmm(x, y.transpose(2,1))
        rx = bdiag(xx).unsqueeze(1).expand(bsx, ny, nx)
        ry = bdiag(yy).unsqueeze(1).expand(bsx, nx, ny)
        P = (rx.transpose(2,1) + ry - 2*zz)
        
        dx = P.min(2)[0] # (bs, nx)
        dy = P.min(1)[0] # (bs, ny)
        
        fscores = []
        for b in range(bsx):
            b_fscores = []
            for i in range(len(thresholds)):
                num = len(np.where(dx <= thresholds[i])[0])
                recall = 100.0 * num / nx
                num = len(np.where(dy <= thresholds[i])[0])
                precision = 100.0 * num / ny
                b_fscores.append((2*precision*recall)/(precision+recall+1e-8))
            fscores.append(b_fscores)
    
        return P.min(2)[0], P.min(1)[0], fscores

def apply_model(df_results):
    # Apply the model 
    #   - 1 image if specified in input
    #   - the set of images of 1 object if only object id specified

    print("="*40)
    if opt.object_id in list_train_object_id:
        print("%s is train" % opt.object_id)
        origin = 'train'
    else:
        print("%s is test" % opt.object_id)
        origin = 'test'
    print("="*40)

    if opt.object_img is None:
        fns_img = sorted(os.listdir(path_id))
        fns_img = [fn for fn in fns_img if ".png" in fn]
    
        for fn in fns_img:
            im = Image.open(os.path.join(path_id, fn))
            im = my_transforms(im) #scale
            img = im[:3,:,:].unsqueeze(0)
             
            img = Variable(img)
            if opt.cuda:
                 img = img.cuda()
             
            #forward pass
            reconstructed_points  = network.forward_inference(img, grid)
             
            #Save output 3D model
            output_fn = os.path.join(output_dir, fn[:-4] + "_" + training + "_")

            write_ply(filename=output_fn + str(int(opt.gen_points)), 
                      points=pd.DataFrame(torch.cat((reconstructed_points.cpu().data.squeeze(), grid_pytorch), 1).numpy()), 
                      as_text=True, text=True, faces=df_faces)

            # Compute Chamfer distance
            n_reconstructed_points =  reconstructed_points.size(1)
            rand_rows = np.random.permutation(n_reconstructed_points)
            reconstructed_points = reconstructed_points[:, rand_rows[:min(opt.chamfer_points, n_reconstructed_points)], :3]
    
            if opt.cuda:
                reconstructed_points = reconstructed_points.type(torch.cuda.FloatTensor)
    
            dist1, dist2, fscores = distChamferFscore(x=true_points, y=reconstructed_points, thresholds=opt.fscore_thresh)
            dist = dist1.mean(1) + dist2.mean(1)

            # Batch size is 1 here
            dist = dist.cpu().detach().numpy()[0]
            fscores = fscores[0]
            
            del reconstructed_points

            row_results = {'training': training, 
                           'paper': 'atlas_net', 
                           'model': opt.model, 
                           'cat': opt.object_cat, 
                           'id': opt.object_id, 
                           'img': fn, 
                           'origin': origin,
                           'chamfer_points': opt.chamfer_points, 
                           'chamfer(x1000)': dist*1000, 
                           'output': fn[:-4] + "_"  + training + "_" + str(opt.gen_points) + ".ply", 
                           'date': get_time()}

            for th, f in zip(opt.fscore_thresh, fscores):
                row_results['fscore(%.1g)' % th] = f
                print("fscore (%.1g) between ground truth and %s/%s is: %.3g" % (th, opt.object_id, fn, f))

            df_results = df_results.append(row_results, ignore_index=True)
            print("Chamfer distance (x1000) between ground truth and %s/%s is: %.3g" % (opt.object_id, fn, dist*1000))
            
        subset = [x for x in df_results.columns if x not in ["chamfer(x1000)", "date"] and "fscore" not in x]
        df_results.drop_duplicates(subset=subset, keep="first", inplace=True)
        df_results.to_csv("../../comparison/df_results.csv", header=True, index=False)
        print("Done transforming %s ! Check out results in comparison/output_atlas_net/" % opt.object_id)
    else:        
        im = Image.open(os.path.join(path_id, opt.object_img))
        im = my_transforms(im) #scale
        img = im[:3,:,:].unsqueeze(0)
        
        img = Variable(img)
        if opt.cuda:
            img = img.cuda()
        
        #forward pass
        reconstructed_points  = network.forward_inference(img, grid)
        
        #Save output 3D model
        output_fn = os.path.join(output_dir, opt.object_img[:-4] + "_" + training + "_")
        points = pd.DataFrame(torch.cat((reconstructed_points.cpu().data.squeeze(), grid_pytorch), 1).numpy())
        write_ply(filename=output_fn + str(int(opt.gen_points)), points=points, as_text=True, text=True, faces=df_faces)
        
        # Compute Chamfer distance
        n_reconstructed_points =  reconstructed_points.size(1)
        rand_rows = np.random.permutation(n_reconstructed_points)
        reconstructed_points = reconstructed_points[:, rand_rows[:min(opt.chamfer_points, n_reconstructed_points)], :3]
        
        if opt.cuda:
            reconstructed_points = reconstructed_points.type(torch.cuda.FloatTensor)
        
        dist1, dist2, fscores = distChamferFscore(true_points, reconstructed_points, thresholds=opt.fscore_thresh)
        dist = dist1.mean(1) + dist2.mean(1)
        
        # Batch size is 1 here
        dist = dist.cpu().detach().numpy()[0]
        fscores = fscores[0]

        del reconstructed_points
        
        row_results = {'training': training, 
                       'paper': 'atlas_net', 
                       'model': opt.model, 
                       'cat': opt.object_cat, 
                       'id': opt.object_id, 
                       'img': opt.object_img, 
                       'origin': origin,
                       'chamfer_points': opt.chamfer_points, 
                       'chamfer(x1000)': dist*1000, 
                       'output': opt.object_img[:-4] +  "_"  + training + "_" + str(opt.gen_points) + ".ply", 
                       'date': get_time()}

        for th, f in zip(opt.fscore_thresh, fscores):
            row_results['fscore(%.1g)' % th] = f
            print("fscore (%.1g) between ground truth and %s/%s is: %.3g" % (th, opt.object_id, opt.object_img, f))

        df_results = df_results.append(row_results, ignore_index=True)
        print("Chamfer distance (x1000) between ground truth and %s/%s is: %.3g" % (opt.object_id, opt.object_img, dist*1000))
        
        subset = [x for x in df_results.columns if x not in ["chamfer(x1000)", "date"] and "fscore" not in x] 
        df_results.drop_duplicates(subset=subset, keep="first", inplace=True)
        df_results.to_csv("../../comparison/df_results.csv", header=True, index=False)    
        print("Done transforming %s/%s ! Check out results in output_atlas_net/" % (opt.object_id, opt.object_img))
        
def object_select_random():
    path_ply = ""
    while not os.path.exists(path_ply):
        object_cat = np.random.choice(list(cat.keys()))
        path_cat = os.path.join("./data/ShapeNet/ShapeNetRendering", cat[object_cat])
        object_id = np.random.choice(os.listdir(path_cat))
        path_id = os.path.join(path_cat, object_id, "rendering")
        path_ply = os.path.join("./data/customShapeNet/", cat[object_cat], "ply", object_id + ".points.ply") 
    print("Randomly selected object %s of category %s (%s)" % (object_id, object_cat, cat[object_cat]))
    return object_cat, path_cat, object_id, path_id, path_ply

#################
# Apply the model
#################

# Load or create csv file used to save results
if not os.path.exists("../../comparison/df_results.csv"):
    df_results = pd.DataFrame(columns=['training', 'paper', 'model', 'cat', 'id', 'img', 'origin', 'chamfer_points',
                                       'chamfer(x1000)', 'fscore(0.001)', 'fscore(0.002)', 'output', 'date'])
else:
    df_results = pd.read_csv("../../comparison/df_results.csv", header='infer')

my_transforms = transforms.Compose([
                 transforms.CenterCrop(127),
                 transforms.Resize(size =  224, interpolation = 2),
                 transforms.ToTensor(),
                 # normalize,
            ])

if opt.object_random == -1:
    # Execute the selected models on all objects found in df_results.csv
    all_ids = set(df_results.id.unique())
    model_ids = set(df_results.loc[df_results.model==opt.model, "id"].unique())

    for object_id in set(all_ids).difference(set(model_ids)):
        opt.object_id = object_id
        opt.object_cat = df_results.loc[df_results.id == object_id].cat.values[0]
        path_cat = os.path.join("./data/ShapeNet/ShapeNetRendering", cat[opt.object_cat])
        path_id = os.path.join(path_cat, opt.object_id, "rendering")
        path_ply = os.path.join("./data/customShapeNet/", cat[opt.object_cat], "ply", opt.object_id + ".points.ply")

        # Recover ground truth model
        ply_set = get_point_cloud(path_ply)
        
        # Ground truth point cloud for Chamfer distance
        # Randomly picks opt.chamfer_points points
        n_ply_points, _ = ply_set.shape
        rand_rows = np.random.permutation(n_ply_points)
        true_points = torch.from_numpy(ply_set[rand_rows[:min(opt.chamfer_points, n_ply_points)], :3])
        true_points = true_points.unsqueeze(0)

        if opt.cuda:
            true_points = true_points.type(torch.cuda.FloatTensor)
    
        # Create output directory if it does not exist already
        output_dir = os.path.join(opt.output_path, cat[opt.object_cat])
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
    
        output_dir = os.path.join(output_dir, opt.object_id)
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
    
        apply_model(df_results)
        df_results = pd.read_csv("../../comparison/df_results.csv", header='infer')
elif not opt.object_random:
    # Recover ground truth model
    path_ply = os.path.join("./data/customShapeNet/", cat[opt.object_cat], "ply", opt.object_id + ".points.ply") 
    ply_set = get_point_cloud(path_ply)
    
    # Ground truth point cloud for Chamfer distance
    # Randomly picks opt.chamfer_points points
    n_ply_points, _ = ply_set.shape
    rand_rows = np.random.permutation(n_ply_points)
    true_points = torch.from_numpy(ply_set[rand_rows[:min(opt.chamfer_points, n_ply_points)], :3])
    true_points = true_points.unsqueeze(0)
    
    if opt.cuda:
        true_points = true_points.type(torch.cuda.FloatTensor)
    
    # Create output directory if it does not exist already
    output_dir = os.path.join(opt.output_path, cat[opt.object_cat])
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    
    output_dir = os.path.join(output_dir, opt.object_id)
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    
    apply_model(df_results)
    df_results = pd.read_csv("../../comparison/df_results.csv", header='infer')
else:
    while opt.object_random > 0:
        # Randomly select an object category and object id
        opt.object_cat, path_cat, opt.object_id, path_id, path_ply = object_select_random()
        
        # Recover ground truth model
        ply_set = get_point_cloud(path_ply)
        
        # Ground truth point cloud for Chamfer distance
        # Randomly picks opt.chamfer_points points
        n_ply_points, _ = ply_set.shape
        rand_rows = np.random.permutation(n_ply_points)
        true_points = torch.from_numpy(ply_set[rand_rows[:min(opt.chamfer_points, n_ply_points)], :3])
        true_points = true_points.unsqueeze(0)

        if opt.cuda:
            true_points = true_points.type(torch.cuda.FloatTensor)
    
        # Create output directory if it does not exist already
        output_dir = os.path.join(opt.output_path, cat[opt.object_cat])
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
    
        output_dir = os.path.join(output_dir, opt.object_id)
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        
        apply_model(df_results)
        df_results = pd.read_csv("../../comparison/df_results.csv", header='infer')
        opt.object_random = opt.object_random - 1 
