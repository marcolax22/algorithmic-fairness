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
    - [Preprocessing Scripts](#preprocessing-scripts)
- [Simulation](#simulation)

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
> It furthermore contrains the functions for the post-selection bias mentioned in Section 3.2 of the thesis.

`fairness_testing.py`
> This file contains the functions for the fairness testing of the datasets, to see the distribution of gender and the corrrelation between the features.

`qualification.py`
> This file contains the functions for the qualification distribution of the different outcomes from the simulation process.

`simulation_vis.py`
> This file contains the distribution visualization functions to see the effects of group fairness interventions and post-selection bias.

## Data

The recruitment dataset is synthetically generated, drawing on a combination of predefined assumptions and historical recruitment records of Dutch graduates. It is specifically designed to simulate graduate-level hiring practices within the Netherlands, particularly in the STEM sector. The dataset contains detailed information on individual applicants, including historical labels indicating past hiring outcomes. Although the data is synthetic, it closely mirrors real-world hiring patterns and decisions, making it a valuable resource for developing and evaluating fairness-aware predictive models.

## Codebook

The codebook contains the codebook for the dataset. It contains the description of the dataset and the variables.

## Graphics

This folder contains the graphics used in the thesis.

## Preprocessing Scripts

The preprocessing scripts are used to preprocess the data for the simulation process.

`data_preprocessing.ipynb`
> This file contains the code for the preprocessing of the data to make the variables compatible for the simulation process.

`fairness_problematic.ipynb`
> This file contains the code for the fairness problematic. It shows the distribution of the gender and the correlation between the features. It uses the functions from the fairness_testing.py file.

`hyperparameter_tuning.ipynb`
> This file contains the code for the hyperparameter tuning of the two models with a validation set of the data.

# Simulation

The simulation is done in the `simulation-model.ipynb` file. It contains the code for the simulation process. It uses the functions from the functions folder. It outputs the results of the described Monte Carlo simulation process in the thesis.