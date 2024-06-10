import os
import numpy as np
from sklearn.model_selection import train_test_split

from PIL import Image
import numpy as np
import pandas as pd


data_dir_deepspace = '/projects/activities/deepspace/team/yue/data/Xenium-data/'
#data_set = 'V1_Adult_Mouse_Brain_Coronal_Section_2'

data_set = 'breast_cancer_sample1_rep1'
#data_set = 'breast_cancer_sample1_rep2'

radius = '128'

#TargetGene = 'POSTN'
#TargetGene = 'IL7R'

##Selected genes representing major cell types are shown: stromal (POSTN), lymphocytes (IL7R), macrophages (ITGAX), myoepithelial (ACTA2, KRT15), endothelial (VWF), DCIS (CEACAM6), and invasive tumor (FASN). 

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--data_id', help='data id for training')
parser.add_argument('--TargetGene', help='Gene')


args, unknown = parser.parse_known_args()

if args.data_id:
    d_id = int(args.data_id)
    data_set = 'breast_cancer_sample1_rep' + str(d_id+1)
    print('data: ')

else:
    data_set = 'breast_cancer_sample1_rep1'
    print('using default rep1 data')


print(data_set)

if args.TargetGene:
    TargetGene = args.TargetGene

else:
    #TargetGene = 'LRRC15'
    TargetGene = 'FASN'

    print('using default Gene')

image_size = 384

def load_data(data_set, TargetGene):
    #STDir="/projects/activities/deepspace/team/elaheh/data/ST/endometrium/SC2200235_EuE-31"#SC2200236_EuE-31
    data_dir = "/projects/li-lab/Yue/DataPool/Spatial/Xenium/"+data_set#SC2200236_EuE-31
    code_dir = '/projects/li-lab/Yue/SpatialAnalysis/'

    print(TargetGene)

    import glob
    import os

    X = []
    Y = []
    voxel_ids = []


    (img_height, img_width) = (image_size, image_size)
    image_sizes = (img_height, img_width)
    input_shape = (img_height, img_width, 3)

    gene_intersect_df = pd.read_csv(code_dir+'py/XeniumExperiments/xenium_genes_intersection.csv', index_col = 0)
    gene_intersect = gene_intersect_df['0']



    if radius == 'one_cell':
        list_of_files = glob.glob(data_dir_deepspace+data_set+"/voxel_pics/*.png")
    elif radius == 'two_cell':
        list_of_files = glob.glob(data_dir_deepspace+data_set+"/voxel_pics_2r/*.png")
    else:
        list_of_files = glob.glob(data_dir_deepspace+data_set+"/voxel_pics_"+radius+"/*.png")


    list_of_files = sorted(list_of_files,
                            key =  lambda x: os.stat(x).st_size)
    ####remove smallest 50 images due since they r blank

    file_sizes = [os.stat(x).st_size for x in list_of_files]


    filtered_files = list_of_files

    if radius == '40':
        filtered_files = list(np.array(list_of_files)[np.array(file_sizes) > 5*1024])
    if radius == '128':
        filtered_files = list(np.array(list_of_files)[np.array(file_sizes) > 20*1024])


    df_Y_origin = pd.read_csv(data_dir_deepspace+data_set+'/spatial_rna.csv', index_col = 0)
    df_Y = df_Y_origin.loc[:,gene_intersect]

    print(df_Y.head())
    
    np.random.seed(0)
    rep_num = 10000
    rand_indx = np.random.choice(160000, rep_num)

    for f in np.array(filtered_files)[rand_indx]:
        #print(f)
        #im = tf.keras.preprocessing.image.load_img(f, target_size=image_sizes)
        #im_array = tf.keras.preprocessing.image.img_to_array(im)
        im_array = np.asarray(Image.open(f).convert('RGB').resize(image_sizes))
        #print(im_array.shape)

        ar = f.split('/')
        voxel_id = int(ar[-1].replace('.png', ''))

        if voxel_id not in df_Y.index:
            continue

        y = list(df_Y.loc[voxel_id,:])

        X.append(im_array)
        Y.append(y)
        voxel_ids.append(str(voxel_id))

    from sklearn.preprocessing import normalize
    #Y = normalize(np.array(Y)+1, axis=1, norm='l1')
    #Y = normalize(np.log(np.array(Y)+1), axis=1, norm='l1')
    Y = np.log(np.array(Y)+1)


    
    Y_filtered = np.array(Y)[:,list(gene_intersect).index(TargetGene)]

    print(np.array(X).shape)
    print(Y_filtered.shape)

    return(X, Y_filtered, voxel_ids)

X, Y_filtered,voxel_ids  = load_data(data_set, TargetGene)

Y_filtered = np.nan_to_num(Y_filtered)
print(np.isnan(Y_filtered).any())

#