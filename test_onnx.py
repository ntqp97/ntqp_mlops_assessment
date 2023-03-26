import numpy as np
from model import ONNXModel, ImagePreprocessor


def test_onnx_model():
    input_size = (224, 224)
    model = ONNXModel()
    preprocessor = ImagePreprocessor(input_size)
    
    image_path1 = './images_test/n01667114_mud_turtle.JPEG'
    expected_class_id = 35
    input_data = preprocessor.preprocess_numpy(image_path1)
    prediction = model.predict(input_data.numpy())
    class_probs = np.exp(prediction)
    predicted_class_id = np.argmax(class_probs)
    assert predicted_class_id == expected_class_id, \
        f"Failed test on image {image_path1}. Expected class id: {expected_class_id}, Predicted class id: {predicted_class_id}"
    print(f"Passed test on image {image_path1}. Predicted class id: {predicted_class_id}")

    image_path2 = './images_test/n01440764_tench.jpeg'
    expected_class_id = 0
    input_data = preprocessor.preprocess_numpy(image_path2)
    prediction = model.predict(input_data.numpy())
    class_probs = np.exp(prediction)
    predicted_class_id = np.argmax(class_probs)
    assert predicted_class_id == expected_class_id, \
        f"Failed test on image {image_path2}. Expected class id: {expected_class_id}, Predicted class id: {predicted_class_id}"
    print(f"Passed test on image {image_path2}. Predicted class id: {predicted_class_id}")
