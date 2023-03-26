import onnxruntime
from torchvision import transforms
from PIL import Image


class ONNXModel:
    def __init__(self, model_path='./models/pytorch_model_weights.onnx'):
        """
        Initialize ONNX model by loading the model with the provided file path
        """
        self.session = onnxruntime.InferenceSession(model_path)

    def predict(self, input_data):
        """
        Make a prediction using the ONNX model and input data
        """
        # get the name of the input and output nodes of the model
        input_name = self.session.get_inputs()[0].name
        output_name = self.session.get_outputs()[0].name
        # run the model with the input data and get the predicted output
        pred = self.session.run([output_name], {input_name: input_data})
        # return the predicted output as a numpy array
        return pred[0]


class ImagePreprocessor:
    def __init__(self, input_size=(224, 224)):
        """
        Initialize image preprocessor with the input size of the model
        """
        self.input_size = input_size


    def preprocess_numpy(self, image_path):
        """
        Preprocess an image using PIL and torchvision transforms
        """
        # Open the image using PIL
        img = Image.open(image_path)

        # Apply image transforms
        resize = transforms.Resize((224, 224))   # Resize to match model input size
        crop = transforms.CenterCrop((224, 224)) # Center crop to match model input size
        to_tensor = transforms.ToTensor()        # Convert PIL image to PyTorch tensor
        normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # Normalize using ImageNet means and std devs
        img = resize(img)
        img = crop(img)
        img = to_tensor(img)
        img = normalize(img)
        img = img.unsqueeze(0)  # Add batch dimension (1, C, H, W)

        # Return the preprocessed image tensor
        return img
