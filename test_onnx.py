import numpy as np
from model import ONNXModel, ImagePreprocessor


def test_onnx_model():
    input_size = (224, 224)
    model = ONNXModel()
    preprocessor = ImagePreprocessor(input_size)

    image_paths = ['./images_test/n01440764_tench.jpeg', './images_test/n01667114_mud_turtle.JPEG']
    expected_class_ids = [0, 35]

    for i, image_path in enumerate(image_paths):
        input_data = preprocessor.preprocess_numpy(image_path)
        prediction = model.predict(input_data.numpy())
        class_probs = np.exp(prediction)
        predicted_class_id = np.argmax(class_probs)
        assert predicted_class_id == expected_class_ids[i], \
            f"Failed test on image {image_path}. Expected class id: {expected_class_ids[i]}, Predicted class id: {predicted_class_id}"
        print(f"Passed test on image {image_path}. Predicted class id: {predicted_class_id}")
