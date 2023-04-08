# Glacier lakes detection using multi-resolution input channels
 Research module at University of Potsdam

The dataset source:

Wilson, R., Glasser, N. F., Reynolds, J. M., Harrison, S., Anacona, P. I., Schaefer, M., & Shannon, S. (2018). Glacial lakes of the Central and Patagonian Andes. Global and Planetary Change, 162, 275-291.

The data was represented as a shapefile with lakes represented as polygones. For creating a dataset that is suitable for machine learning purposes, the data from 2016 year was taken. The corresponding satellite images were manually downloaded. 

Data preprocessing folder contains code used for the following purposes:

- combining bands for Landsat images;
- creating labels based on the shapefiles;
- overlapping the images with the mask of the glacier;
- cropping images and labels into patches of 1024x1024 size. 
