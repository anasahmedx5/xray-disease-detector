# Chest X-Ray Pneumonia Detection Using MobileNetV2

![Chest X-Ray Example](assets/person100_bacteria_480.jpeg)

---

## **Project Overview**

This project implements a **deep learning model for detecting pneumonia from chest X-ray images** using **Transfer Learning with MobileNetV2**. The model is trained on the **Chest X-Ray Pneumonia dataset** from Kaggle, which contains labeled images of **Normal** and **Pneumonia** cases.  

The main goal of this project is to allow a user to **upload a chest X-ray image** and receive:

- **Prediction:** Pneumonia or Normal  
- **Confidence Score:** Probability of disease  

This project demonstrates the integration of **deep learning** with a **user-friendly web interface**, making it suitable for portfolios, academic projects, or AI healthcare demos.

---

## **Technologies and Libraries**

- **Python 3.9+**  
- **TensorFlow / Keras** – Deep Learning framework  
- **MobileNetV2** – Pretrained Transfer Learning model  
- **OpenCV** – Image preprocessing  
- **Matplotlib** – Data visualization  
- **Streamlit** – Web app for interactive interface  
- **NumPy / scikit-learn** – Data handling and preprocessing  

---

## **Dataset**

The project uses the **Chest X-Ray Pneumonia Dataset** from Kaggle:

- **Train:** 5216 images (Normal + Pneumonia)  
- **Validation:** 16% of the dataset  
- **Test:** 624 images  

Dataset link: [Chest X-Ray Pneumonia Dataset](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia)

---

## **Model Architecture**

The model uses **MobileNetV2** as the base network with **pretrained ImageNet weights**. A custom classification head is added:

1. **GlobalAveragePooling2D** – Reduces feature map dimensions  
2. **BatchNormalization** – Stabilizes and accelerates training  
3. **Dense Layer (128 units, ReLU)** – Learns complex patterns  
4. **Dropout (0.5)** – Reduces overfitting  
5. **Dense Layer (1 unit, Sigmoid)** – Outputs probability of Pneumonia  

**Why MobileNetV2?**

- Lightweight and efficient  
- High accuracy on small datasets  
- Ideal for web deployment  

---

## **Training Details**

- **Image Preprocessing:** Rescale pixels [0,1], data augmentation (rotation, zoom, horizontal flip, shift)  
- **Loss Function:** Binary Crossentropy  
- **Optimizer:** Adam (learning rate = 1e-4)  
- **Batch Size:** 32  
- **Epochs:** 12  
- **Class Imbalance Handling:** Using `class_weight` in Keras  

---

## **Performance**

| Metric | Value |
|--------|-------|
| Test Accuracy | ~90–94% |
| Loss | ~0.15–0.2 |
| Regularization | BatchNormalization + Dropout |

> **Note:** Accuracy may vary slightly depending on GPU, dataset splits, and random seed.

---

## **Web App / Streamlit Integration**

The model can be deployed as a **Streamlit web application**, where users can:

- Upload their **chest X-ray image**  
- Get **prediction**: Normal or Pneumonia  
- See **confidence score** (probability)

**Streamlit Advantages:**

- Quick deployment and prototyping  
- Interactive and responsive UI  
- Easy sharing of AI applications  

---

## **Future Features**

- Grad-CAM heatmap visualization to show **which areas influenced the prediction**  
- Two-column modern interface with **uploaded X-ray and Grad-CAM overlay**  
- Additional metrics like **sensitivity, specificity, and confusion matrix**  
