***

# Stacked Specialist Regressors (SSR)

Student entry for the COCO Keypoints Challenge 2016.

Team : 
**C. Sutton, R. Canyasse, G. Demonet**

Abstract :
**TODO**

Table of contents :

+ [Introduction](#introduction)
+ [Installation](#install)
+ [Quick start](#quickstart)
+ [The idea behind](#idea)
+ [Some potential improvements](#improvements)
+ [References](#references)

<a name="introduction"></a>
## Introduction

TODO

<a name="install"></a>
## Installation

TODO

<a name="quickstart"></a>
## Quick start

TODO

<a name="idea"></a>
## The idea behind

TODO

<a name="improvements"></a>
## Some potential improvements

TODO

<a name="references"></a>
## References

TODO

***
***
***

GitHub release to-do list :

- [x] Prepare project build structure
- [ ] Clean and adapt code
- [ ] Write new necessary code for sharing (experiment reproducibility, results...)
- [ ] Document the reproduction and skeleton of the code
- [ ] Complete README with background and potential leads forward

***

**1. _Project build structure_**

Several components :

+ **Data acquisition and handling**
    + Links to MS COCO dataset and challenge
    + Script to automatically populate the data folders if empty
    + Image batches generators for Keras models
    
+ **Person detection task : Faster-RCNN (thanks to [github.com/rbgirshick](https://github.com/rbgirshick/py-faster-rcnn))**
    + Model description and links
    + Script to generate bounding boxes for test images
    
+ **Keypoints regression task : SSR model**
    + Model definition : keras implementation
    + Training script
    + Prediction script
    
+ **Evaluation and results**
    + Model output handling + "aggregation" method
    + Visualizations
    + Graphs and performance/time/space reports
    
The repo structure is then :

+ **data/**
    + **coco/**
    + **scripts/**
    + _README.md_
    
+ **frcnn/**
    + **???/**
    + _README.md_
    
+ **ssr/**
    + **model/**
    + _README.md_
    
+ **eval/**
    + **scripts/**
    + _README.md_
    
+ _README.md_