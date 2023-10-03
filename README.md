# Computer Vision Notebooks Collection

Hey there! 👋 Welcome to my collection of Jupyter Notebooks focused on computer vision tasks. In this repository, you'll find notebooks covering a wide range of topics, including image classification, face recognition, and more. We'll explore various deep learning architectures and techniques to tackle these exciting computer vision challenges.

**Table of Contents**
- [Getting Started](#getting_started) 
- [Models description](#model_description)
    - [Image Classifitcaion](#classification)
    - [Object Detection](#detection)
    - [Face Recognition](#recognition)
    - [Image captioning](#captioning)

## What's Inside?

This repository is a treasure trove of computer vision knowledge. You'll discover:

- Implementations of popular architectures like VGG, ResNet, and more for image classification.
- Notebooks dedicated to face recognition using state-of-the-art techniques.
- Experiments with object detection, semantic segmentation, and other vision tasks.
- Tutorials on data preprocessing, augmentation, and model evaluation.
- Real-world applications and case studies showcasing the power of computer vision.

## Prerequisites

Before you get started, make sure you have the following:

- Python installed (Python 3.x is recommended).
- Jupyter Notebook installed (if not, you can install it with pip).
- Required libraries like PyTorch, Transformers which are in the requirements.txt file.
- A curious mind and a passion for computer vision!

## <a name="getting_started"> Getting Started</a>
1. Clone or download this repository to your local machine.

   ```bash
   git clone https://github.com/zucchini-nlp/computer-vision
    ```
2. Open the Jupyter Notebook in your favorite environment (e.g., Jupyter Notebook, JupyterLab).
3. Follow along with the notebook's instructions and code blocks.
4. Experiment, tweak, and modify the code to see how different changes affect the model's performance.
5. Have fun and learn!

## <a name="model_description">Models descriptions</a>


### <a name="classification">Image Classification</a>

# Image Classification

Image classification is a fundamental computer vision task that involves training machines to recognize and categorize objects or patterns within images. This capability is at the core of various applications, from identifying objects in photos to enabling self-driving cars to detect road signs and pedestrians.

In this repository, I've provided three notebooks, each offering a unique approach to image classification. These notebooks serve as practical examples to help you grasp the concepts and gain hands-on experience in image classification. You'll find demonstrations using different architectures and libraries, allowing you to explore various techniques and methodologies.

**Explore the Notebooks**

Here's a quick overview of the three notebooks available:

1. **Notebook 1: VGG from Scratch on CIFAR100 with PyTorch** - This notebook guides you through the process of building a VGG (Visual Geometry Group) model from scratch using the PyTorch framework. You'll apply this model to the CIFAR-100 dataset, gaining hands-on experience in creating a deep neural network for image classification. Explore this notebook if you're interested in building neural networks from the ground up.

2. **Notebook 2: ResNet from Scratch with PyTorch Lightning** - In this notebook, you'll explore the construction of a ResNet model from scratch using PyTorch Lightning. ResNet is a popular architecture known for its deep network design. You'll learn how to implement and train this model for image classification tasks. If you're curious about building and training deep convolutional networks, this notebook is for you.

3. **Notebook 3: Fine-tuning Vision Transformer (ViT) with Hugging Face Trainer** - The third notebook takes a different approach by introducing the Vision Transformer (ViT) architecture. You'll utilize the Hugging Face Trainer framework to fine-tune a pre-trained ViT model for image classification, showcasing the power of transfer learning. This notebook is perfect for those interested in leveraging pre-trained models for image classification tasks.


### <a name="detection">Object Detection</a>



### <a name="recognition">Face Recognition</a>

Face recognition is a specialized field within computer vision that involves identifying and verifying individuals based on their facial features. It's a crucial component of biometric security systems and has a wide range of applications, from unlocking your smartphone to enhancing surveillance and access control.

**Notebook Highlights: Face Recognition with ResNet from Scratch**

- **Dataset:** The notebook utilizes the "Labelled Faces in the Wild" dataset, a collection of diverse labeled face images. This dataset provides a rich resource for training and evaluating your face recognition model with real-world face data.

- **Model:** You'll learn how to construct a ResNet-based face recognition model from scratch. ResNet is renowned for its deep network design, making it well-suited for the complex task of recognizing faces in varying conditions.

- **Loss Functions:** The notebook explains two crucial loss functions: Triplet Loss and Contrastive Loss. These functions play a pivotal role in training an effective face recognition model. Triplet Loss encourages the model to reduce the distance between embeddings of matching faces while increasing the distance between non-matching faces, and Contrastive Loss enforces a similar concept.

### <a name="captioning">Image Captioning</a>




