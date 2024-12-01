
[![linting: pylint](https://img.shields.io/badge/linting-pylint-yellowgreen)](https://github.com/pylint-dev/pylint)

## Summary

### Repo Introduction

This repo implements and performs an experiment between **ResNet** model and **Plain CNN model** which are described in the paper of  *Deep Residual Learning for Image Recognition*.

The implementation is straight-forward and beginner-friendly. Python files are organized into the standard sequential steps of learning:
 i. data augmentation
 ii. model definition of ResNet and Plain CNN using `Keras`
 iii. training scripts
 iv. automatic model analysis
 v. prediction examples
 
 ### Quick Peeks

(0) Sampling from training dataset.
<div style="display: flex; justify-content: center;">
  <img src="pic/cooked-img-sample.jpg" alt="resnet" style="width: 70%;">
</div>

(1) For Plain CNN, 34 layer is worse than its 18 layer version. While for ResNet, 34 layer is better. And comparably speaking, ResNet is way better than Plain CNN models for both 18 layer model and 34 layer model.
<div style="display: flex; justify-content: space-between;">
  <img src="pic/expr-1-plainet.png" alt="plainet" style="width: 45%;">
  <img src="pic/expr-1-resnet.png" alt="resnet" style="width: 45%;">
</div>
<div style="display: flex; justify-content: center;">
  <img src="pic/expr-1-compare.png" alt="resnet" style="width: 60%;">
</div>

(2) Some predictions for unseen data.
<div style="display: flex; justify-content: center;">
  <img src="pic/predictions.jpg" alt="resnet" style="width: 70%;">
</div>
