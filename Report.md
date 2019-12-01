# Image Recognition: Andrea Abellera and Andrew Marinic
 ## Introduction
<p>	The goal of our project is to create an algorithm for  image recognising. We will be testing a series of images containing balls and other containing round objects that are not balls. These known objects can be used to help us refine our search. </p>
	 <p>	The algorithm  will use the uncanny edge detection to reduce our test image to its edges. The number of trials can be modified to help refine the search. A single trial would search the image as a whole, and then each subsequent trial would scan the trial image to help identify smaller balls, multiple balls, and off centre balls. The algorithm was calculates the eigen matrix of the image library and compares the difference between the matrix and the test image or slice.  We then perform a similar test comparing the difference between our individual library images and the test image to see if we have a good match to any of our library images. The stretch goal would be to extend this methodology to track the progress of the maze mosaic virus in a plant cell. Throughout the report it will be mentioned how certain functions were coded with this in mind.</p> 

## Hypothesis
<p> Our hypothesis is as follows. If we reduce an image down to its edges with the uncanny edge detection this will allow us to gather the most defining aspects of an object, while eliminating less important information. Comparison of images that have not been reduced to its edges would leave far too many variables to give fair comparisons. At this stage in our algorithm the image is reduced to it's edges, and so are the images in our library. In our situation a series of functions  to generate a directory with perfectly circular rings of varying radii. We will then use a similar process to the facial recognition method discussed in class ** [reference] **. As we would like to extend this application to the maze mosaic virus we chose to design the algorithm with the ability to scan. The shape in which we scan depends on what we are looking for. We will either scan as a square or a rectangle. An eigein matrix is calculated from the library of reference images, this is used to test how close an image is to resembling a single ball in the slice (which can be the entire image). We will then test each slice against each individual image in our reference library. Performing these two tests will allow for us to see how much our image varies from our reference library and if it is acceptable which image in the library is it most like. The reference images are named after the radius of the circle in the image allowing for us to estimate the balls approximate pixel radius. This allows us to extend the application to testing the maze cell for infected organelles vs non infected organelles, as we would need to pass a reference of healthy vs non healthy regions of the cell and compare how different from each of the libraries it is so we can associate the region as healthy or infected. Our ball application is essentially the same with "has a ball" and "does not have a ball" as our replacements for healthy and unhealthy. </p>

## Methodology
 Using uncanny edge detection is essential for this project.

 - choosing a generated library and comparing balls and non balls to it as opposed to choosing images to compare directly to may prevent bias?
 - 


<!--stackedit_data:
eyJoaXN0b3J5IjpbLTEyNTM4MDY3NDEsLTgzNjc3Njg4NCw1NT
Q1NTE4OTgsLTE3OTI1MDQ1MTksLTcyNzc1NDg4MSwxMzM5OTIx
MTgzXX0=
-->