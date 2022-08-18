# Computer Vision Assignment 2
This assignment is composed of 5 questions, the code of each is contained in its own python file. The outline to run the code is discussed below.


## Question 1
The first question contains two examples of the apply_homography() function, one was provided, and the other I wrote as a learning exercise. The Functions accept either a numpy array corresponding to image data, or the image itself. The second argument to the function is the matrix corresponding to the transformation to be applied. To run the code, simply read the image using Image.open() as shown in the code, get an array from one of the get_<type>_array() functions, and execute the apply_homography() function with these as arugments.

## Question 2
To execute Question 2, it is as easy as just running the python file. The file consists of a call to the apply_homography() function mentioned before, as well as a merge() function which stitches the poster image onto the target building image.

## Question 3
To execute this question, it is also very straight forward, just run the python file. All outputs should be either shown to screen, or written to the output/ directory.

## Question 4
There are 2 pasting options, if you simply run the python file, it will execute the first pasting option, as described in my report. If you want to do the second pasting option, simply remove the exit() call on line 119, and rerun the program.

## Question 5
Question 5 contains many output images, so it is easiest to simply run one of the following provided functions. 

```python
     draw_mapping(i), draw_ransac_mappings(i), recompute_H_ALL(i)
```
Where i is the number corresponding to the file to be processed. These functions draw all SIFT matches, draw the filtered SIFT matches, and draw the transformed bounding box, respectively.
