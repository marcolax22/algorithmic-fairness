# Impact of Algorithmic Group Fairness Intervention on Hiring Decisions
<p> Marco Lax <br>
Thesis submitted on: 02.06.2025 </p>

---

## Table of contents:
- [Summary](#summary)
- [Executing the code](#executing-the-code)
- [Folder Structure](#folder-structure)
    - [Functions](#functions)
    - [Data](#data)
    - [Graphics](#graphics)
- [Citation](#citation)

# Summary 

This repository contains the code for the thesis _"The Impact of Algorithmic Group Fairness Intervention on
Hiring Decisions – A Simulation Study"_. This code is written to reproduce the results of the thesis. It contains all graphics, functions and data used in the thesis.

# Executing the code

Use pip install for the project requirements:

`pip install -r requirements.txt`

Used Python Version: `Python3.13.3`

# Folder Structure

The code is structered in different folders:

## Functions

The functions folder contains the functions used in the code.

`algorithmic_model.py`
> This file contains the code for the logistic regression model and random forest model.

`fairness_testing.py`
> This f

`simulation_vis.py`
> This file contains the distribution visualization functions to see the effects of group fairness interventions and post-selection bias.

## Data

The recruitment dataset is synthetically generated, drawing on a combination of predefined assumptions and historical recruitment records of Dutch graduates. It is specifically designed to simulate graduate-level hiring practices within the Netherlands, particularly in the STEM sector. The dataset contains detailed information on individual applicants, including historical labels indicating past hiring outcomes. Although the data is synthetic, it closely mirrors real-world hiring patterns and decisions, making it a valuable resource for developing and evaluating fairness-aware predictive models.

## Graphics

This folder contains the graphics used in the thesis.

# Citation