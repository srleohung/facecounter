#!/bin/bash
# Reference from https://github.com/legolas123/cv-tricks.com
OPENPOSE_URL="http://posefs1.perception.cs.cmu.edu/OpenPose/models/"
POSE_FOLDER="pose/"

# MPI pretrained model
MPI_FOLDER=${POSE_FOLDER}"data/mpi/"
MPI_MODEL=${MPI_FOLDER}"pose_iter_160000.caffemodel"
wget -c ${OPENPOSE_URL}${MPI_MODEL} -P ${MPI_FOLDER}

# Body_25 pretrained model
BODY_25_FOLDER=${POSE_FOLDER}"data/body_25/"
BODY_25_MODEL=${BODY_25_FOLDER}"pose_iter_584000.caffemodel"
wget -c ${OPENPOSE_URL}${BODY_25_MODEL} -P ${BODY_25_FOLDER}

# COCO pretrained model
COCO_FOLDER=${POSE_FOLDER}"data/coco/"
COCO_MODEL=${COCO_FOLDER}"pose_iter_440000.caffemodel"
wget -c ${OPENPOSE_URL}${COCO_MODEL} -P ${COCO_FOLDER}

