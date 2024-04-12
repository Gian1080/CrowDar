# CrowDar

A Convolutional Neural Network to detect crow-like birds!

## Introduction

Welcome to my birdwatching paradise—or as I like to call it, the CrowCam Project! Like many others, I find immense joy in feeding birds on my balcony, particularly the clever and mischievous crows and jackdaws. There’s something uniquely charming about these feathered intellectuals that captivates the heart of any bird lover.

However, there's a small hitch in my birdwatching saga. Every time I excitedly whip open my curtains to check on my avian friends, they scatter in a flurry of panic. It seems my enthusiasm is a bit too much for these cautious creatures. To solve this problem without missing out on their delightful antics, I've turned to technology for a solution.

Enter CrowDar! It's a simple yet brilliant setup involving a webcam that keeps an eye on the balcony. With a bit of coding magic and some machine learning, this system not only lets me watch these birds undisturbed but also sends me a nifty notification whenever a crow-like bird is spotted. This way, I can observe their behavior, see what treats they prefer, and enjoy my morning coffee without causing a feathered frenzy.

So, whether you're a fellow bird enthusiast, a tech geek, or just here for some light reading, I hope you find this project both amusing and inspiring. Let’s dive into the world of backyard ornithology and smart technology combined, all from the comfort of our screens!

## Technologies

The CrowDar Project harnesses a variety of technologies to bring the fascinating world of crows and jackdaws right to your screen, without startling them away. Here’s a breakdown of the tech stack that makes it all possible:

- **Python**: The backbone of our project, Python's versatility allows us to handle everything from image processing to running machine learning models seamlessly.
- **TensorFlow and Keras**: These powerful libraries are at the heart of our machine learning operations, enabling us to train models that can identify different bird species from webcam images.
- **PIL (Python Imaging Library)**: We use PIL to load and manipulate images easily, which is essential for preparing our data sets.
- **NumPy**: This library supports our data handling needs, making it easier to perform operations on large arrays and matrices of numerical data—a fundamental aspect of machine learning.
- **Matplotlib**: A favorite for anyone who needs to visualize data, we use Matplotlib to plot and understand the images and annotations from our dataset.
- **JSON**: Used to handle data interchange, JSON files in the COCO format store annotations related to the images in our dataset, enabling efficient data management and model training.
- **COCO Dataset**: Leveraged for training the model, the Common Objects in Context (COCO) dataset provides a rich set of images and annotations for object detection, specifically tailored to enable detailed image analysis and object localization.

Together, these technologies not only power the technical aspects of detecting and observing birds but also make the project accessible for further development and user interaction. Whether you're tweaking the model or simply enjoying birdwatching, these tools ensure a smooth and enjoyable experience.


## Installation

To set up this project locally and start observing the crows and jackdaws, follow these steps:

1. **Clone the repository:**
   Clone the project repository by running the following command in your terminal:
   ```bash
   cd MyDestinationFolder
   git clone https://github.com/Gian1080/CrowDar.git
   cd CrowDar

2. **Create a Virtual Environment (Recommended):**
   To avoid conflicts with other Python packages, it's a good practice to create a virtual environment. Use the following commands to create and activate a virtual environment:

   For Windows:
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```
   For macOS and Linux:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
3. **Install Dependencies:**
Once the virtual environment is activated, install the required Python packages specified in the requirements.txt file:

   ```bash
   pip install -r requirements.txt
   ```
   Run the Application:
   After installing all dependencies, run the application using:

   ```bash
   python app.py
   ```
   Replace app.py with the actual entry file of your project if it's different.

4. **Deactivate the Virtual Environment:**
When you are done, you can deactivate the virtual environment by simply running: