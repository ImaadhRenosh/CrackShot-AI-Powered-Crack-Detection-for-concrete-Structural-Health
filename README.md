# IBM AI engineering Professional Certified Capstone Project : AI-powered Concrete Crack Detection System

## Table of Contents
- [Overview](#overview)
- [Objective](#objective)
- [Technologies Used](#technologies-used)
- [Prerequisites](#prerequisites)
- [Installation & How to Run](#installation--how-to-run)
- [Demo](#demo)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)
- [Contact](#contact)

# Overview
The AI-powered Concrete Crack Detection System, developed as part of the AI Capstone Project with Deep-Learning (IBM AI Engineering Specialization), is designed to ensure the structural health of concrete structures. Crack detection holds vital importance for structural health monitoring and inspection. By leveraging advanced machine learning techniques, this project aims to provide accurate and efficient crack detection solutions. It uses a pre-trained ResNet18 model fine-tuned with PyTorch to classify concrete images into positive (cracked) and negative (non-cracked) samples, achieving a validation accuracy of 99.43%.

# Project Aim
The aim of this project is to develop a robust AI-powered system for detecting cracks in concrete structures by downloading and preprocessing the Concrete dataset, exploring the data through visualization, fine-tuning a pre-trained ResNet18 model for binary classification (cracked vs. non-cracked), and evaluating its performance through loss analysis and misclassification identification to ensure effective structural health monitoring.

## Technologies Used
- **Programming Language:** Python
- **Libraries:** TensorFlow, OpenCV, numpy
- **Tools:** Jupyter Notebook, Git, VS Code

## Prerequisites
- Python 3.x
- TensorFlow
- OpenCV
- numpy

## Installation & How to Run
To set up and run the project locally:

1. **Clone the Repository**:
    ```sh
    git clone https://github.com/ImaadhRenosh/CrackShot-AI-Powered-Crack-Detection-for-concrete-Structural-Health.git
    cd CrackShot-AI-Powered-Crack-Detection-for-concrete-Structural-Health
    ```

2. **Install Dependencies**: Install the required libraries using:
    ```python
    import os

    # Download the data 
    !wget https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DL0321EN/data/images/Positive_tensors.zip 

    ! wget https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DL0321EN/data/images/Negative_tensors.zip

    !unzip -qo Positive_tensors.zip 

    !unzip -qo Negative_tensors.zip

    note : I will Download the cocncrete images dataset and unzip the files in the data directory, unlike the other projects,
    all the data will be deleted after the project is closed, the download may may take some time:

    
    
    # Install required libraries
    !pip install torchvision

    # Import Required Libraries
    # These are the libraries will be used for this Project.
    import torchvision.models as models
    from PIL import Image
    import pandas
    from torchvision import transforms
    import torch.nn as nn
    import time
    import torch 
    import matplotlib.pylab as plt
    import numpy as np
    from torch.utils.data import Dataset, DataLoader
    import h5py
    import os
    import glob
    torch.manual_seed(0)

    from matplotlib.pyplot import imshow
    import matplotlib.pylab as plt
    from PIL import Image
    import pandas as pd
    import os
    ```

3. **Run the Jupyter Notebook**: Launch Jupyter Notebook and open `Crack_Detection.ipynb`.

## Demo

### Create a Dataset Class :

<img width="899" alt="Screenshot 2025-04-05 at 04 39 44" src="https://github.com/user-attachments/assets/24ff2ea9-8e58-4538-b636-d9394b06625f" />

<img width="895" alt="Screenshot 2025-04-05 at 04 40 20" src="https://github.com/user-attachments/assets/a6e5bbee-e50d-4177-8adc-7accef5fb074" />

<img width="903" alt="Screenshot 2025-04-05 at 04 40 47" src="https://github.com/user-attachments/assets/50df945c-5903-4988-a71e-8db459397752" />



### Part 1 : I prepare a model, and change the output layer

<img width="980" alt="Screenshot 2025-04-05 at 04 45 58" src="https://github.com/user-attachments/assets/fad917d9-27d9-4c6b-9741-6de80fc3ae0e" />

<img width="893" alt="Screenshot 2025-04-05 at 04 47 57" src="https://github.com/user-attachments/assets/f8e77ca8-6d20-476c-9598-911b4bfc3fa2" />

<img width="767" alt="Screenshot 2025-04-05 at 04 48 36" src="https://github.com/user-attachments/assets/ba08eb74-8d32-48ba-bf22-85656a4fd468" />

<img width="793" alt="Screenshot 2025-04-05 at 04 49 20" src="https://github.com/user-attachments/assets/c1f7567b-df5b-477d-8bca-fbab08562ca4" />



### Part 2: Train the Model

<img width="915" alt="Screenshot 2025-04-05 at 04 50 41" src="https://github.com/user-attachments/assets/df482c77-914f-4779-8bd1-d030c912ea77" />

<img width="895" alt="Screenshot 2025-04-05 at 04 51 54" src="https://github.com/user-attachments/assets/f1af4970-36c5-4b99-9ea2-e841c451bbca" />

<img width="896" alt="Screenshot 2025-04-05 at 04 52 28" src="https://github.com/user-attachments/assets/ea4e8144-60ab-437a-92c0-fbb6613d3a39" />

<img width="901" alt="Screenshot 2025-04-05 at 04 52 59" src="https://github.com/user-attachments/assets/dc0e69f1-0c71-4a61-bed0-c67baa811dd0" />


### Part 3 : Find 4 misclassified samples

<img width="899" alt="Screenshot 2025-04-05 at 04 54 51" src="https://github.com/user-attachments/assets/362a4a44-b746-4cda-8e24-9e3f61398eef" />

<img width="895" alt="Screenshot 2025-04-05 at 04 55 42" src="https://github.com/user-attachments/assets/5b36135b-a2cf-4cb9-a201-f9b4960cf912" />

<img width="879" alt="Screenshot 2025-04-05 at 04 56 12" src="https://github.com/user-attachments/assets/d5046692-6eb3-493b-96cc-1628006a9894" />

<img width="908" alt="Screenshot 2025-04-05 at 04 56 47" src="https://github.com/user-attachments/assets/2011bd04-04b9-4e43-874a-6661c583560b" />

<img width="882" alt="Screenshot 2025-04-05 at 04 57 08" src="https://github.com/user-attachments/assets/8caa0686-9e99-4715-84d4-ae61fa9aef57" />

<img width="832" alt="Screenshot 2025-04-05 at 04 57 19" src="https://github.com/user-attachments/assets/5700eef8-6909-4502-a31c-4406fa432236" />



## Usage
After launching the Jupyter Notebook, you can:
- Import crack detection data.
- Train the AI model.
- Detect cracks in new images.

## Contributing
Contributions are welcome! If you’d like to improve this project, please follow these steps:
1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Commit your changes and push the branch.
4. Open a pull request detailing your changes.

## License
This project is licensed under © IBM Corporation. All rights reserved.

## Acknowledgements
- Thanks to the contributors of TensorFlow, OpenCV, and numpy.
- Special thanks to the structural health monitoring community for their support and data.

## Contact
For any questions or suggestions, feel free to reach out:
- **Email:** imaadhrenosh@gmail.com
- **LinkedIn profile**: [LinkedIn profile](https://www.linkedin.com/in/imaadh-renosh-007aba348)
