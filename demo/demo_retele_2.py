import os
from PIL import Image
import numpy as np
import torch
from capsulenet import CapsuleNet
from torchvision import transforms


class ML_Runner:

    def __init__(self):

        self.models_dict = {'capsule': {'FF': r'E:\saved_model\Capsule\FF\capsule_fullface_epoch_4_param_FF_186_1851.pkl',
                                        'celebDF': r'E:\saved_model\Capsule\celebDF\capsule_fullface_epoch_6_param_celebDF_196_328.pkl'},
                            'resnet-50': {'FF': r'E:\saved_model\ResNet\ff\2\resnet-50_fullface_epoch_6_param_FF_276_1035.pkl',
                                          'celebDF': r'E:\saved_model\ResNet\celebDF\2\resnet-50_fullface_epoch_7_param_celebDF_276_340.pkl'},
                            'Xception': {'FF': r'E:\saved_model\Xception\FF\Xception_fullface_epoch_6_param_FF_166_1424.pkl',
                                         'celebDF': r'E:\saved_model\Xception\celebDF\Xception_fullface_epoch_4_param_celebDF_166_2114.pkl'},
                            'capsule-LSTM': {'celebDF': r'E:\saved_model\Capsule-LSTM\capsule-LSTM_fullface_epoch_3_param_celebDF_206_021.pkl'},
                            'resnet-50-LSTM': {'FF': r'E:\saved_model\ResNet-LSTM\FF\resnet-50-LSTM_fullface_epoch_4_param_FF_146_2358.pkl',
                                               'celebDF': r'E:\saved_model\ResNet-LSTM\celebDF\resnet-50-LSTM_fullface_epoch_6_param_celebDF_156_712.pkl'},
                            'Xception-LSTM': {'FF': r'E:\saved_model\Xception-LSTM\FF_raw_fullface\fullface_epoch_12_param_2018_5615.pkl',
                                              'celebDF': r'E:\saved_model\Xception-LSTM\CelebDF_fullface\fullface_epoch_3_param_celebDF_194_1853.pkl'}
                            }

        self.device = None
        self.model = None

    def run_capsulenet(self, image, model_type='capsule', dataset='FF', transform=None, multiple_frames=False):

        if self.model is None or self.device is None:

            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            self.model = CapsuleNet(architecture=model_type, dataset=dataset)
            self.model.to(self.device)
            self.model.load_state_dict(torch.load(self.models_dict[model_type][dataset]))

        if not multiple_frames:
            data = np.array(image, dtype=np.float32)
            if np.max(data) > 1:
                data = data / 255
            data = transform(data)
        else:
            data = torch.stack([transform(x) for x in image])

        self.model.eval()
        with torch.no_grad():
            if not multiple_frames:
                data = torch.stack([data, torch.zeros(data.shape)]).to(self.device)
            else:
                data = data.to(self.device)

            out = self.model(data)

        return out


if __name__ == "__main__":


    folder_to_predict = 'id50_id52_0001' #'id50_id56_0001' #'id50_id54_0001' #'id50_id52_0001' #'id50_0001'

    img_folder = 'F:\saved_celefdf_all'
    images = os.listdir(os.path.join(img_folder, folder_to_predict))
    print(len(images), 'frames!')

    transf_imagenet = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        # Resnet and VGG19 expects to have data normalized this way (because pretrained)
    ])

    runner = ML_Runner()
    probs = []
    i = 0
    for img in images:

        img_adr = os.path.join(img_folder, folder_to_predict, img)
        uploaded_image = Image.open(img_adr)

        prob = runner.run_capsulenet(uploaded_image, model_type='capsule',
                                     dataset='celebDF', transform=transf_imagenet,
                                     multiple_frames=False)
        probs.append(prob.cpu().detach()[0])

        i += 1
        if i % 10 == 0:
            print(i, '/', len(images))

    print('\n\n\n\n')

    prob_df = torch.Tensor(probs).mean().item()
    print(prob_df * 100, '% deepfake probability!')

    if prob_df > 0.5:
        print('Video IS deepfake!')
    else:
        print('Video is NOT deepfake')