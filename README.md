# 3D Data Processing course
This repo contains my submitted homeworks from the 3D Data Processing course at UniPD - DEI 2022/23.

## Homework 1 - Stereo Matching

Search in a pair of stereo images for corresponding pixels that are the projections of the same 3D point, considering the epipolar constraint that reduces the search to one dimension.

Implement the **Semi-Global Matching algorithm** presented during the lecture.

A description of the assignment can be found [here](https://github.com/stefanobinotto/3D-Data-Processing-course/blob/main/lab1/Slide%203DP%20Lab1.pdf), while [here](https://github.com/stefanobinotto/3D-Data-Processing-course/blob/main/lab1/Report_Lab1_BinottoStefano.pdf) is the report showing the approach and the results.

Assessment: 100/100

## Homework 2 - Structure from Motion

Estimate the 3D structure of a small scene taken by your smartphone from a sequence of images with some field-of-view overlaps.

A description of the assignment can be found [here](https://github.com/stefanobinotto/3D-Data-Processing-course/blob/main/lab2/Lab2%20-%20Structure%20from%20Motion.pdf), while [here](https://github.com/stefanobinotto/3D-Data-Processing-course/blob/main/lab2/Report3DDPLab2.pdf) is the report showing the approach and the results.

Assessment: 92/100

## Homework 3 - Iterative Closest Point Cloud Registration

Given a source and a target point cloud roughly aligned, find the fine alignment transformation of the source to the target cloud.

A description of the assignment can be found [here](https://github.com/stefanobinotto/3D-Data-Processing-course/blob/main/lab3/Lab3%20-%20Cloud%20Registration.pdf), while [here](https://github.com/stefanobinotto/3D-Data-Processing-course/blob/main/lab3/StefanoBinotto_Lab3.pdf) is the report showing the approach and the results.

Assessment: 100/100

## Homework 4 - Deep 3D descriptors

Design a modified PointNet architecture that is able to extract 3D descriptors to be used for matching.

The goal of this assignment is to design a reduced version of the PointNet architecture we call here **TinyPointNet** that learns a 3D local feature descriptor from training data. The
idea is to use the n-dimensional global feature learned by TinyPointNet directly as a descriptor of a locality of points (neighborhood of a point p), removing from PointNet the last
MLP (multi-layer perceptron) used for classification tasks. 

A description of the assignment can be found [here](https://github.com/stefanobinotto/3D-Data-Processing-course/blob/main/lab4/Lab4%20-%20Deep%203D%20descriptors.pdf), while [here](https://github.com/stefanobinotto/3D-Data-Processing-course/blob/main/lab4/Stefano_Binotto_report_lab4.pdf) is the report showing the approach and the results.

Assessment: 100/100

## Poster Presentation

Public presentation of a scientific paper to other interested students, PhD students, or researchers from DEI.

[Paper](https://github.com/stefanobinotto/3D-Data-Processing-course/blob/main/poster%20presentation/Zhang_Not_All_Points_Are_Equal_Learning_Highly_Efficient_Point-Based_Detectors_CVPR_2022_paper.pdf) | [Poster](https://github.com/stefanobinotto/3D-Data-Processing-course/blob/main/poster%20presentation/Poster_Presentation.pdf) | Assessment: 100/100
