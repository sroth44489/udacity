import pandas as pd
import numpy as np
import time
import pickle
from scipy import spatial

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as manimation

from os import listdir
from os.path import isfile, join

import ply_file as ply

def load_class_id(class1_points_file_name, class2_points_file_name, num_point_cloud_points, kdtree):
    """
    This function loads the hand labeled data and returns a class_id label for the point
    cloud points. The data for the entire point cloud is stored in kdtree. The hand
    segmented point clouds are stored in the files class1_points_file_name and 
    class2_points_file_name. These files contain only the points belonging to
    the label class1 and class2.

    Parameters
    -------------------
    class1_points_file_name: string
        This is the path to the class 1 point cloud data

    class2_points_file_name: string
        This is the path to the class 2 point cloud data
    
    num_point_cloud_points: int
        The number of points in the full point cloud that is stored in kdtree

    kdtree: KD Tree
        This is the kd tree used to find the point in the point cloud that
        is closest to the hand labeled point. It contains information on the entire
        point cloud

    Return Values
    -------------------
    class_id: np.array
        The class ID of each point in the point cloud. The index in the array
        corresponds to the index of the point cloud point
    """

    # Read in the class 1 data points
    data = ply.read_ply_file(class1_points_file_name)

    class_points = np.array(data[['x', 'y', 'z']])

    # Create the class id array
    class_id = np.zeros(num_point_cloud_points, dtype=np.int)

    # For each point, find its index in the point cloud and set its class_id
    for query_point in class_points:
        point_distance, point_index = kdtree.query(query_point, k=1, p=np.inf)
        class_id[point_index] = 1

    # Classify the class 2 points if they are there
    if class2_points_file_name is not None:
        data = ply.read_ply_file(class2_points_file_name)
        class_points = np.array(data[['x', 'y', 'z']])
        for query_point in class_points:
            point_distance, point_index = kdtree.query(query_point, k=1, p=np.inf)
            class_id[point_index] = 2

    return class_id

def load_training_data(point_cloud_file_name, class1_points_file_name, class2_points_file_name=None):
    """
    This function loads the training data. The training data consists of
    the point cloud data, and a set of hand classified points. For each
    point in the point cloud, its features are calculated and it is
    given a class_id identifying its hand classified object type. The hand
    classified points are labeled as class 1 and (optionally) class 2

    Parameters
    -------------------
    point_coud_file_name: string
        point cloud file name

    class1_points_file_name: string
        point cloud file name containing the class 1 points

    class2_points_file_name: string
        point cloud file name containing the class 2 points

    Return Values
    -------------------
    data: pandas DataFrame
        The training data. The point cloud data is stored in columns
        'x','y','z','intensity' and the class label is stored in
        'class_id'

    """

    # Load the point cloud data
    data = ply.read_ply_file(point_cloud_file_name)

    point_cloud_points = np.array(data[['x', 'y', 'z']])
    point_cloud_intensities = np.array(data['intensity'])

    # Build a KD tree to make finding nearest neighbors faster.
    #Use only x,y,z points (not intensity) to build tree
    kdtree = spatial.cKDTree(point_cloud_points)

    # Calculate the point cloud features
    data = calculate_features(point_cloud_points, point_cloud_intensities, kdtree)

    # Load the hand labeled class data
    class_id = load_class_id(class1_points_file_name, class2_points_file_name,
                             point_cloud_points.shape[0], kdtree)
    data['class_id'] = pd.Series(class_id, index=data.index)

    return data

def calculate_features(point_cloud_points, point_cloud_intensities, kdtree, pca_min_radius=0.06):
    """
    This function calculates the geometric and directional features of a point cloud.
    Instead, the features describing a point must come from a region of the point cloud.
    Given the size of object being classified and the resolution of the flash lidar,
    a 6cm cube is an appropriate volume. However, because points within the point cloud
    are not evenly distributed, the size of the cube must be adapted especially in areas
    of low point density (Zakhor, 2011). In the case where insufficient points are located
    in a 6cm cube, the size of the cube is increased until a minimum number of points (15)
    are included. This is to ensure that a sufficient number of points are used to generate
    accurate statistics of the volume.
    Once a representative group of points is gathered, features describing the local geometry
    are calculated. The features used are common in analysis of point clouds. They use the eigen
    values and vectors of the covariance matrix in the region around a point. Given the eigen
    values w_3<w_2<w_1 of the covariance matrix, the geometric features are
    {pointness = w_1, surfaceness = w_2-2_1, linearness = w_3-w_2} These geometric features
    represent point-ness, surface-ness and linear-ness of the region. In addition, the algorithm
    contains directional features using the local tangent and normal vectors. The tangent and
    normal vectors are estimated using the eigen vectors of the largest and smallest eigen
    values. The sine and cosine of these vectors {v_t,v_n} with respect to the horizontal
    plane are used, giving a total of 4 directional features. To estimate the confidence
    in these features, the features are scaled according to the strengths of their
    corresponding eigen values:
    scale{v_t,v_n}={linearness,surfaceness }/(max(linearness,pointness,surfaceness)). 
    The complete feature vector concatenates the 3 geometric features and 4 directional
    features for a resulting 7D feature vector.

    Parameters
    -------------------
    point_cloud_points: np.array
        2D array of point cloud points. Each row contains an (x,y,z) point

    point_cloud_intensities: np.array
        1D array of point cloud intensities.

    kdtree: spatial.cKDTree
        kd tree used to find points in a region

    pca_min_radius: float
        the radius of the neighborhood to use when calculating geometric and
        directional features

    Return Values
    -------------------
    features: pandas.DataFrame
        The first 4 columns are the x,y,z,intensity data from the point cloud.
        The remaining columns are the geometric and directional features for that point.

    """

    # To prevent areas of great density from bogging down teh algorithm, set a max
    # number of points to use in pca.
    pca_max_points = 200
    # To prevent areas of low density from impacting statistical significance of pca,
    # put a floor on the number of points used in pca.
    pca_min_points = 15

    # Define the names of hte features found
    feature_names = ["x", "y", "z", "intensity", "pointness", "surfaceness", "linearness",
                     "cos_tangent", "sin_tangent", "cos_normal", "sin_normal"]
    # Features is a 2D matrix. Each row is the features of a point in the point cloud
    features = np.empty([point_cloud_points.shape[0], len(feature_names)])

    # For each point in the point cloud, calculate a feature
    feature_index = 0
    for query_point in point_cloud_points:
        points_distance, points_index = kdtree.query(query_point, k=pca_max_points,
                                                     distance_upper_bound=pca_min_radius, p=np.inf)
        # if we don't get enough points to do pca, expand the radius to
        # ensure you get a minimum number of points
        num_points = np.count_nonzero(np.isfinite(points_distance))
        if num_points < pca_min_points:
            points_distance, points_index = kdtree.query(query_point, k=pca_min_points)
        # Get a list of the points (with dist < inf)
        points = [point_cloud_points[index]
                  for dist, index in zip(points_distance, points_index) if np.isfinite(dist)]
        points = np.transpose(np.array(points))
        # points is now a matrix with columns of (x,y,z) triplets

        # Find the covariance of the points
        cov = np.cov(points)

        # Find the eigen values(ascending), vectors of the covariance matrix
        w, v = np.linalg.eigh(cov)

        # Spectral features from Munoz icra 2009
        pointness = w[0]
        surfaceness = w[1] - w[0]
        linearness = w[2] - w[1]
        max_spectral_feature = max([pointness, surfaceness, linearness])
        # Directional Features from Munoz icra 2009
        tangent = v[2]
        normal = v[0]

        # Find the sine and cosine of the tangent line w.r.t. horizontal (x,y) plane
        adjacent = np.sqrt(tangent[0]**2 + tangent[1]**2)
        opposite = tangent[2]
        hypotenuse = np.sqrt(adjacent**2 + opposite**2)
        cos_tangent = adjacent / hypotenuse
        sin_tangent = opposite / hypotenuse
        # scale the values based on strength of the extracted directions
        scale = linearness / max_spectral_feature
        cos_tangent *= scale
        sin_tangent *= scale
        # Find the sine and cosine of the normal line w.r.t. horizontal (x,y) plane
        adjacent = np.sqrt(normal[0]**2 + normal[1]**2)
        opposite = normal[2]
        hypotenuse = np.sqrt(adjacent**2 + opposite**2)
        cos_normal = adjacent / hypotenuse
        sin_normal = opposite / hypotenuse
        # scale the values based on strength of the extracted directions
        scale = surfaceness / max_spectral_feature
        cos_normal *= scale
        sin_normal *= scale

        # create the new feature vector
        new_feature = [points[0][0], points[1][0], points[2][0],
                       point_cloud_intensities[points_index[0]],
                       pointness, surfaceness, linearness, cos_tangent,
                       sin_tangent, cos_normal, sin_normal]

        features[feature_index, :] = new_feature
        feature_index += 1

    data = pd.DataFrame(data=features, columns=feature_names)
    return data

def calculate_point_cloud_features(file_name, pca_min_radius=0.06):
    """
    This function loads in a point cloud file and calculates the features
    of each point cloud point

    Parameters
    -------------------
    file_name: string
        name of the ply file

    pca_min_radius: float
        radius of the neighborhood used to calculate the point features

    Return Values
    -------------------
    data: pandas.DataFrame
        data frame with the x,y,z,intensity, feature values

    """

    data = ply.read_ply_file(file_name)

    point_cloud_points = np.array(data[['x', 'y', 'z']])
    point_cloud_intensities = np.array(data['intensity'])

    # Build a KD tree to make finding nearest neighbors faster.
    # Use only x,y,z points (not intensity) to build tree
    kdtree = spatial.cKDTree(point_cloud_points)

    data = calculate_features(point_cloud_points, point_cloud_intensities, kdtree,
                              pca_min_radius=pca_min_radius)
    return data

def animate_classifier(data_file_path, classifier_save_file_name):
    """
    This function creates an animation of the classifier in action.
    It loads all of the ply files on the given path, classifies the points
    and makes a movie of the classified plots.
    The animation is saved to the file 'writer_test.mp4'

    Parameters
    -------------------
    data_file_path: string
        path to the directory with the ply files

    classifier_save_file_name: string
        pickle file name used to save the classifier

    Return Values
    -------------------
    None.

    """

    clf = pickle.load(open(classifier_save_file_name, "rb"))

    FFMpegWriter = manimation.writers['ffmpeg']
    metadata = dict(title='Movie Test', artist='Matplotlib',
                    comment='Movie support!')
    writer = FFMpegWriter(fps=15, metadata=metadata)

    fig = plt.figure()
    cane_plt, = plt.plot([], [], 'r.')
    cordon_plt, = plt.plot([], [], 'b.')
    plt.xlim(-1.0, 1.0)
    plt.ylim(-1.0, 1.0)
    file_text = plt.text(-0.5, -0.875, '')

    with writer.saving(fig, "writer_test.mp4", 100):
        only_files = [f for f in listdir(data_file_path) if isfile(join(data_file_path, f))]
        ply_files = [f for f in only_files if f[-9:] == '.flt4.ply']
        ply_files.sort()
        # ply_files = [ply_files[index] for index in range(30)]

        for point_cloud_file_name in ply_files:
            point_cloud_extension = '.flt4.ply'
            # Only animate the point cloud files. Do not animate the cordon point files, etc.
            if point_cloud_file_name[-len(point_cloud_extension):] == point_cloud_extension:
                data = calculate_point_cloud_features(data_file_path+point_cloud_file_name)
                # Remove the point cloud x,y,z data from the data frame. Those will not be
                # used as features
                point_cloud_points = data[['x', 'y', 'z', 'intensity']]

                del data['x']
                del data['y']
                del data['z']
                del data['intensity']

                X_all = data
                predicted_class = clf.predict(X_all)

                classified_point_cloud = point_cloud_points[['x', 'y', 'z']]
                classified_point_cloud['class_id'] = predicted_class

                cordon_points = classified_point_cloud.loc[classified_point_cloud['class_id'] == 1]
                cane_points = classified_point_cloud.loc[classified_point_cloud['class_id'] == 0]

                cordon_plt.set_data(-cordon_points['z'], -cordon_points['y'])
                cane_plt.set_data(-cane_points['z'], -cane_points['y'])
                file_text.set_text(point_cloud_file_name)
                writer.grab_frame()

def test_calculate_point_cloud_features():
    """
    This function tests the calculate_point_cloud_features function
    """

    # Data file from rosbag play 2017-01-18-13-50-16.bag
    file_name = "/media/sroth/adata/gallo/cordonTracking/data/ply/1484765416981987000.flt4.ply"
    tstart = time.time()
    data = calculate_point_cloud_features(file_name)
    print ("execution time = ", time.time() - tstart)
    print data.head
    # scatter_matrix(data['eigVal1', 'eigVal2', 'eigVal3'])

def test_load_training_data():
    """
    This function tests the load_training_data function
    """
    point_cloud_file_name = \
        "/media/sroth/adata/gallo/cordonTracking/data/ply/1484765416981987000.flt4.ply"
    cordon_points_file_name = \
        "/media/sroth/adata/gallo/cordonTracking/data/ply/1484765416981987000-Cordon.ply"
    data = load_training_data(point_cloud_file_name, cordon_points_file_name)
    print data

#test_load_training_data()
#animate_classifier("/media/sroth/adata/gallo/cordonTracking/data/ply/", "/media/sroth/adata/gallo/cordonTracking/src/classifier.p")