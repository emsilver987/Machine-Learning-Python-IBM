# Machine Learning with Python (IBM - Coursera)

Welcome to my personal repository for the [Machine Learning with Python](https://www.coursera.org/learn/machine-learning-with-python) course by IBM on Coursera. I created this space to organize my course materials, including notebooks, code examples, and additional resources, as I work through the exciting world of machine learning using Python.

## Table of Contents

- [Overview](#overview)
- [Course Modules](#course-modules)
- [Final Project](#final-project)
- [What I've Learned](#what-ive-learned)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Directory Structure](#directory-structure)
- [Resources](#resources)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Overview

I enrolled in this course to build a solid understanding of machine learning concepts and practical applications using Python. Through a series of lectures, hands-on assignments, and interactive Jupyter notebooks, I am learning about both supervised and unsupervised learning techniques. This repository serves as a living document of my journey, capturing my notes, experiments, and reflections as I progress through the material.

## Course Modules

The course is divided into several modules, and here’s a quick overview of each:

- **Introduction to Machine Learning:**  
  I received an overview of machine learning fundamentals, its applications, and the basic tools required for building models.

- **Data Preparation and Analysis:**  
  This module covered techniques for cleaning, preprocessing, and analyzing data—crucial steps before feeding data into machine learning algorithms.

- **Supervised Learning:**  
  I explored regression, classification, and various evaluation metrics to understand how models are built and validated.

- **Unsupervised Learning:**  
  Here, I learned about clustering, dimensionality reduction, and pattern discovery, which are useful when working with unlabeled data.

- **Practical Applications:**  
  This part of the course allowed me to apply what I learned to real-world problems through case studies and project-based learning.

Each module is organized in its own folder within this repository.

## Final Project

As a capstone project for this course, I built a machine learning pipeline to predict whether it will rain today in Melbourne, Australia, using historical weather data. Here's a high-level summary of what I implemented:

- **Data Source:**  
  Weather data from multiple Melbourne locations, cleaned and filtered for relevant entries.

- **Feature Engineering:**  
  Mapped dates to seasons, renamed columns for clarity, and separated numerical and categorical features.

- **Preprocessing Pipeline:**  
  Used `ColumnTransformer` to scale numeric features and one-hot encode categorical ones. Handled unknown categories and ensured the pipeline could be reused for both training and inference.

- **Model Training & Tuning:**  
  Built and tuned a `RandomForestClassifier` and `LogisticRegression` using `GridSearchCV` with 5-fold cross-validation and stratified splits. Optimized for accuracy and balanced class weights due to class imbalance.

- **Evaluation:**  
  Reported accuracy, precision, recall, and F1-score. Displayed confusion matrices using both `matplotlib` and `seaborn` for visual analysis. Identified top contributing features to model predictions using feature importance scores.

This project gave me valuable hands-on experience in handling imbalanced classification problems, designing preprocessing workflows, evaluating multiple models, and visualizing performance.

## What I've Learned

So far, the course has helped me:
- Understand the core principles of machine learning and its various applications.
- Get hands-on experience with Python libraries like NumPy, pandas, matplotlib, and scikit-learn.
- Learn the importance of data cleaning and preparation before model training.
- Build, evaluate, and fine-tune predictive models.
- Appreciate the role of both supervised and unsupervised learning in solving different types of problems.
- Build end-to-end pipelines for real-world classification tasks.

This journey has laid a strong foundation for my continued exploration of machine learning.

## Prerequisites

Before diving into this course, it's helpful to have:
- A basic understanding of Python programming.
- Familiarity with libraries such as NumPy, pandas, matplotlib, and scikit-learn.
- An environment capable of running Jupyter Notebooks (e.g., Anaconda or a virtual environment with the necessary packages).

## Installation

To set up your environment, follow these steps:

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/emsilver987/Python-IBM-Project
   cd Python-IBM-Project
   ```

2. **(Optional but recommended) Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate        # On Windows use: venv\Scripts\activate
   ```

3. **Install the required packages:**
   ```bash
   pip install -r requirements.txt
   ```