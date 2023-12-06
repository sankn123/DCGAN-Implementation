import torch
import os
import cv2
from model import Generator, Discriminator
import torch.nn as nn
import numpy as np


# Loading dataset
dataset_path='DCGAN_from_scratch/Dataset/img_align_celeba'
train_real=[]
train_real_labels=[]
for img in os.listdir(dataset_path):

    img_path= os.path.join( dataset_path,img )
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # cropping it to a size of 64x64 in the center
    original_height, original_width = image.shape[:2]
    start_y = max(0, (original_height - 64) // 2)
    end_y = start_y + 64
    start_x = max(0, (original_width - 64) // 2)
    end_x = start_x + 64
    cropped_image = image[start_y:end_y, start_x:end_x]
    train_real.append(cropped_image)
    train_real_labels.append(1)

train_real=torch.FloatTensor(train_real)
train_real_labels=torch.LongTensor(train_real_labels)

train_real=torch.permute(train_real,(0,3,1,2))

batch_norm=nn.BatchNorm2d(3)
train_real=batch_norm(train_real)
# loading the models
generator=Generator()
discriminator=Discriminator()

opt1=torch.optim.Adam(generator.parameters(),0.0002) #generator
opt2=torch.optim.Adam(discriminator.parameters(),0.0002) #discriminator

criterion = nn.CrossEntropyLoss()

# initializing with normal weights
for name,para in generator.named_parameters():
    if 'weight' in name:
        nn.init.normal_(para,0, 0.02)
    elif 'bias' in name:
        nn.init.constant_(para,0)

k=train_real.shape[0]
epochs=100
print("<<<<<training started>>>>>")
for epoch in range(epochs):
    generator_input=[]
    fake_labels=[]
    for _ in range(k):
        white_noise=np.random.normal(0,1,100)
        generator_input.append(white_noise)
        fake_labels.append(0)
    
    generator_input=np.array(generator_input)

    generator_input=torch.FloatTensor(generator_input)
    fake_labels=torch.LongTensor(fake_labels)

   
    #feeding to generator
    preds_generator=generator(generator_input)

    # creating a tensor of fake and real images
    dicriminator_input=torch.concat([preds_generator,train_real])
    discriminator_labels=torch.concat([fake_labels,train_real_labels])

    dicriminator_input=torch.FloatTensor(dicriminator_input)
    #shuffeling them 
    num_samples = dicriminator_input.size(0)
    indices = torch.randperm(num_samples)
    dicriminator_input = dicriminator_input[indices]
    discriminator_labels = discriminator_labels[indices]

    #feeding to discriminator 
    preds_discriminator=discriminator(dicriminator_input)

    #calculating the loss
    loss_discriminator=criterion(preds_discriminator,discriminator_labels)

    #backprop for discriminator
    opt2.zero_grad()

    loss_discriminator.backward(retain_graph=True)
    opt2.step()
    

    ## updaitng generator weights ##

    generator_input2=[]
    fake_labels2=[] # These will be equal to 1(label of real image), as we want to update gnerator to produce images close to real images.
    for _ in range(k):
        white_noise=np.random.normal(0,1,100)
        generator_input2.append(white_noise)
        fake_labels2.append(1)

    generator_input2=np.array(generator_input2)

    generator_input2=torch.FloatTensor(generator_input2)
    fake_labels2=torch.LongTensor(fake_labels2)

    preds_generator2=generator(generator_input2)

    # passing generator output through discriminator 
    preds_discriminator2=discriminator(preds_generator2)

    loss_discriminator2=criterion(preds_discriminator2,fake_labels2)

    #backprop for generator
    opt1.zero_grad()
    loss_discriminator2.backward(retain_graph=True)
    opt1.step()
    
    

    print(f' Epoch {epoch+1}: discriminator loss (discriminator) {loss_discriminator} discriminator loss (generator) {loss_discriminator2}')

    # Saving a generated image after 50 epochs
    if epoch+1==17:
        white_noise=np.random.normal(0,1,100)
        white_noise=torch.FloatTensor(white_noise)
        white_noise=torch.unsqueeze(white_noise,0)
        pred=generator(white_noise)
        pred=torch.squeeze(pred)
        
        pred=torch.permute(pred,(1,2,0))
        from PIL import Image

        image_array = (pred * 255).byte().numpy()
        image_pil = Image.fromarray(image_array, 'RGB')
        image_pil.save('output.jpg')


    



