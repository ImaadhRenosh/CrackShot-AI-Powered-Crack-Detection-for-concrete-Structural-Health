# CrackShot-AI-Powered-Crack-Detection-for-concrete-Structural-Health

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

## Overview
CrackShot is an AI-powered crack detection system designed to ensure the structural health of concrete structures. Crack detection holds vital importance for structural health monitoring and inspection. By leveraging advanced machine learning techniques, CrackShot aims to provide accurate and efficient crack detection solutions.

## Objective
- How to download and pre-process the Concrete dataset.
Crack detection has vital importance for structural health monitoring and inspection. We aim to train a network to detect cracks, where images containing cracks are positive, and images without cracks are negative. This involves downloading the data, studying the dataset, and plotting a few images.

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

Create a Dataset Class :
<img width="899" alt="Screenshot 2025-04-05 at 04 39 44" src="https://github.com/user-attachments/assets/24ff2ea9-8e58-4538-b636-d9394b06625f" />

<img width="895" alt="Screenshot 2025-04-05 at 04 40 20" src="https://github.com/user-attachments/assets/a6e5bbee-e50d-4177-8adc-7accef5fb074" />

<img width="903" alt="Screenshot 2025-04-05 at 04 40 47" src="https://github.com/user-attachments/assets/50df945c-5903-4988-a71e-8db459397752" />







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
