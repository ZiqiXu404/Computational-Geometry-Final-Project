# Computational Geometry Final Project - Sweepline Algorithm & Segment Tree
*Stephanie Xu*

*CS163 Computational Geometry, Fall 2022*

*Professor Diane Souvaine*

## Overview
Inspired by a leetcode problem, I applied the vertical line sweep algorithm we learned this semster to compute the skyline of a set of rectangular buildings that are projected onto a plane. The skyline is the upper contour of the rectangular buildings that are built on the same flat ground.

I also implemented a segment tree data structure to handle queries about the number of buildings on a line, which is the number of buildings overlapped at a given point on the projected plane.

The program allows user's input to randomly generate a set of buildings, visualize the buildlings and the skyline, and query about the "overlapping" buildings.

## To use

* ##### External dependencies
    ```pip install matplotlib```
    
    ```pip install numpy```
* Run the program

    ```python sweepline.py```

## Files
* sweepline.py
* sweepline.ipynb
* README.md

## References
* CS163 Lecture notes
* Leetcode the skyline problem: https://leetcode.com/problems/the-skyline-problem/
