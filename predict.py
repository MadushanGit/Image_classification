from PIL import Image
import numpy as np
import train
import matplotlib.pyplot as plt
import torch
import argparse
from torchvision import datasets, models, transforms
from torch.autograd import Variable
import random
import os

def arg_parser():
    # Define parser
    parser = argparse.ArgumentParser(description="Neural Network Settings")
    
    parser.add_argument('--category_names', 
                        type=str, 
                        default="cat_to_name.json",
                        help='Name of the flowers: cat_to_name.json')
    
    parser.add_argument('--load_dir', 
                        type=str,
                        default="checkpoint.pth",
                        help='Load directory')
        
    parser.add_argument('--image', 
                        type=str,
                        help='Image file path for the prediction.',
                        required=True)
        
    parser.add_argument('--top_k', 
                        type=int, 
                         default=5,
                        help='Choose top K matches as int.')

    parser.add_argument('--gpu', 
                        action="store_true", 
                        default="gpu",
                        help='Use GPU')
    # Parse args
    args = parser.parse_args()
    return args
    
args = arg_parser()    

with open(args.category_names, 'r') as f:
    cat_to_name = json.load(f)

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = models.chekpoint["archi"](pretrained=True)
    model.classifier = checkpoint["classifier"]
    model.epochs = checkpoint["epochs"],
    model.class_to_idx = checkpoint["class_to_idx"],
    model.optimizer = checkpoint["optimizer"],
    model.load_state_dict(checkpoint["state_dict"])
    
    return model

load_checkpoint(filepath=args.load_dir)

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    with Image.open(image) as im:
        im = im.resize((256,256))
        dimension = 0.5*(256-224)
        im = im.crop((dimension,dimension,256-dimension,256-dimension))
        im = np.array(im)/255

        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        im = (im - mean) / std

        return im.transpose(2,0,1)
    
def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax


#     plt.imshow(image)
    
def predict(image_path, model, topk=args.top_k):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    
    cuda = torch.cuda.is_available()
    if cuda & args.gpu == "gpu":
        model.cuda()
    else:
        model.cpu()
    
    # turn off dropout
    model.eval()

    image = process_image(image_path=args.image)
   
    image = torch.from_numpy(np.array([image])).float()
    
    image = Variable(image)
    if cuda & args.gpu:
        image = image.cuda()
        
    output = model.forward(image)
    
    probs= torch.exp(output).data
    
  
    prob = torch.topk(probs, topk)[0].tolist()[0] 
    index = torch.topk(probs, topk)[1].tolist()[0] 
    
    ind = []
    for i in range(len(model.class_to_idx.items())):
        ind.append(list(model.class_to_idx.items())[i][0])

    label = []
    for i in range(5):
        label.append(ind[index[i]])

    return prob, label

# img = random.choice(os.listdir('./flowers/test/100/'))
# img_path = './flowers/test/100/' + img
# with  Image.open(img_path) as image:
#     plt.imshow(image)
    
prob, classes = predict(args.image, model)
print(prob)
print(classes)
print([cat_to_name[x] for x in classes])

prob, classes = predict(args.image, model)
max_index = np.argmax(prob)
max_probability = prob[max_index]
label = classes[max_index]

fig = plt.figure(figsize=(8,8))
ax1 = plt.subplot2grid((15,9), (0,0), colspan=9, rowspan=9)
ax2 = plt.subplot2grid((15,9), (9,2), colspan=5, rowspan=7)

image = Image.open(img_path=args.image)
ax1.axis('off')
ax1.set_title(cat_to_name[label])
ax1.imshow(image)

labels = []
for i in classes:
    labels.append(cat_to_name[i])
    
y_position = np.arange(5)
ax2.set_yticks(y_position)
ax2.set_yticklabels(labels)
ax2.set_xlabel('Probability')
ax2.invert_yaxis()
ax2.barh(y_position, prob, xerr=0, align='center', color='blue')

plt.show()