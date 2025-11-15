# End-to-End Fraud Detection ML Pipeline

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![Azure ML](https://img.shields.io/badge/Service-Azure%20ML-0078D4?logo=microsoftazure)](https://azure.microsoft.com/en-us/services/machine-learning/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 🚀 Overview

This repository hosts a complete **End-to-End Machine Learning Pipeline** for **fraud detection**. The entire workflow, from data ingestion and exploratory analysis to model training, deployment, and operationalization, is built and orchestrated using **Azure Machine Learning (Azure ML) Service**.

The goal of this project is to demonstrate best practices for MLOps by automating the complete lifecycle of a high-impact, real-world machine learning application.

## 🎯 Key Features

* **Azure ML Integration:** Full pipeline orchestration (data, training, deployment) using Azure ML compute and services.
* **Data Science Workflow:** Dedicated stages for data sourcing, EDA, feature engineering, and modeling.
* **Modular Code Structure:** Organized into distinct folders representing phases of the ML lifecycle.
* **Fraud Detection Model:** Trains a robust classification model to identify fraudulent transactions.

## ⚙️ Project Architecture & Pipeline Stages

The repository is structured around the sequential stages of the machine learning pipeline:

| Folder | Description |
| :--- | :--- |
| `azure_connection` | Contains configuration and scripts for connecting to and setting up the **Azure ML Workspace**. |
| `data` | Placeholder for raw and processed datasets, including scripts for data acquisition and cleaning. |
| `exploratory_data_analysis` | Jupyter notebooks and scripts used for initial data understanding, visualization, and feature engineering. |
| `train` | Scripts for training the fraud detection model, logging metrics, and registering the final model in Azure ML. |
| `deploy` | Contains necessary files (e.g., scoring script, environment configuration) for packaging the model and deploying it as a **real-time inference endpoint** on Azure Kubernetes Service (AKS) or Azure Container Instance (ACI). |

***

## 🛠️ Technologies Used

This project leverages the following key technologies and libraries:

| Category | Technology | Purpose |
| :--- | :--- | :--- |
| **Cloud Platform** | **Azure Machine Learning Service** | MLOps orchestration, compute management, model registry, and endpoint deployment. |
| **Language** | Python | Core programming language for all ML steps. |
| **Data Analysis** | Jupyter Notebooks | Interactive development and exploratory data analysis. |
| **Libraries** | `pandas`, `numpy`, `scikit-learn` | Data manipulation, numerical operations, and machine learning algorithms. |
| **MLOps** | Azure ML SDK | Interacting with the Azure ML Workspace and defining pipeline components. |

***

## 🚀 Setup & Installation

Follow these steps to set up the project and connect to your Azure environment.

### 1. Prerequisites

* An active **Azure Subscription**.
* An **Azure Machine Learning Workspace** configured.
* The **Azure CLI** installed and configured for your subscription.
* Python 3.8+ installed.

### 2. Clone the Repository

```bash
git clone [https://github.com/CodeTemka/e2e-fraud-detection.git](https://github.com/CodeTemka/e2e-fraud-detection.git)
cd e2e-fraud-detection
