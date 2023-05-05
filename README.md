# Bean Leaf Classification

The goal of this project is to create a model that can accurately detect and classify diseases in bean leaves using a dataset of images, which can help farmers identify and treat diseased plants more quickly and effectively. ResNet50, MobileNetV3, and VGG19 is used to train the model.
## Dataset

The [iBean](https://github.com/AI-Lab-Makerere/ibean/) dataset was used for training and evaluating the model. It consists of 1295 images of bean plants with annotations for various plant diseases and health conditions. The dataset was divided into training , validation, and testing sets and resized to 224x224.

## Methods

Three different deep learning models were used to classify the bean leaf diseases: ResNet50, MobileNetV3, and VGG19. These models were trained on the training set using categorical cross-entropy loss and optimized using the Adam optimizer. The model with the lowest validation loss was selected as the final model and evaluated on the testing set.

## Results and Discussion

The performance of the three models on the testing set is shown in the following table:

| Model         | Accuracy | Precision | Recall  | F1 Score |
|---------------|----------|-----------|---------|----------|
| ResNet50      | 91.49    | 91.71     | 91.41   | 91.45    |
| MobileNetV3   | 92.19    | 92.51     | 92.19   | 92.24    |
| VGG19         | 93.73    | 93.77     | 93.75   | 93.70    |

As shown in the table, VGG19 achieved the highest accuracy, precision, recall, and F1 score among the three models. This can be attributed to the deeper architecture of VGG19, which allows it to learn more complex features from the images. However, the difference in performance between VGG19 and the other two models is relatively small.

## Conclusion

In conclusion, this project demonstrates the effectiveness of deep learning models in classifying bean leaf diseases. VGG19 achieved the highest performance among the three models, but the other two models, ResNet50 and MobileNetV3, also achieved good results. This project can be extended by using more advanced architectures, such as EfficientNet or ResNeSt, or by using transfer learning techniques to fine-tune pre-trained models on this dataset.
