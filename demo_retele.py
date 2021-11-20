import streamlit as st
from PIL import Image
import numpy as np
import base64
import os
import torch
from capsulenet import CapsuleNet
from torchvision import transforms
from cnn import CNN
import skvideo.io
import copy

skvideo.setFFmpegPath('C:\ProgramData\Anaconda3\Lib\site-packages\skvideo\io')

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

    def run_capsulenet(self, image, model_type='capsule', dataset='FF', transform=None, multiple_frames=False):

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = CapsuleNet(architecture=model_type, dataset=dataset)
        model.to(device)
        model.load_state_dict(torch.load(self.models_dict[model_type][dataset]))

        if not multiple_frames:
            data = np.array(image, dtype=np.float32)
            if np.max(data) > 1:
                data = data / 255
            data = transform(data)
        else:
            data = torch.stack([transform(x) for x in image])

        model.eval()
        with torch.no_grad():
            if not multiple_frames:
                data = torch.stack([data, torch.zeros(data.shape)]).to(device)
            else:
                data = data.to(device)

            out = model(data)

        return out

    def run_cnn(self, image, model_type='resnet-50', dataset='FF', transform=None, multiple_frames=False):

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = CNN(architecture=model_type)
        model.to(device)
        model.load_state_dict(torch.load(self.models_dict[model_type][dataset]))

        if not multiple_frames:
            data = np.array(image, dtype=np.float32)
            if np.max(data) > 1:
                data = data / 255
            data = transform(data)
        else:
            data = torch.stack([transform(x) for x in image])

        model.eval()
        with torch.no_grad():
            if not multiple_frames:
                data = torch.stack([data, torch.zeros(data.shape)]).to(device)
            else:
                data = data.to(device)

            out = model(data)

        return out

    def run_cnnlstm(self, frames, model_type='resnet-50-LSTM', dataset='FF', transform=None):

        frames_in = copy.deepcopy(frames)
        if model_type == 'Xception-LSTM':
            from cnnlstm import CNNLSTM
            model = CNNLSTM()
        elif model_type == 'resnet-50-LSTM':
            from cnnlstm_resnet import CNNLSTM
            model = CNNLSTM()
        elif model_type == 'capsule-LSTM':
            model = CapsuleNet(architecture=model_type)
            frames_in = frames_in[::3] #Need 20 frames for this architecture
        else:
            raise ModuleNotFoundError('Model type not defined')

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.load_state_dict(torch.load(self.models_dict[model_type][dataset]))

        data = torch.stack([transform(x) for x in frames_in])

        model.eval()
        with torch.no_grad():
            data = torch.stack([data, data]).to(device)
            out = model(data)

        return out.detach()


    def isdeepfake(self, prob, type='image',  lth=0.35, hth=0.65):

        if type == 'image':
            text_not = '<p style="font-family:Modern; color:Green; font-size: 22px; background-color: White; display: inline-block;">' \
                    'Image is <b>NOT</b> a DeepFake!</p>'
            text_yes = '<p style="font-family:Modern; color:Red; font-size: 22px; background-color:White; display: inline-block;">' \
                    'Image <b>IS</b> a DeepFake!</p>'
            text_maybe = '<p style="font-family:Modern; color:Orange; font-size: 22px; background-color:White; display: inline-block;">' \
                    'Image may be a DeepFake!</p>'
        elif type == 'video':
            text_not = '<p style="font-family:Modern; color:Green; font-size: 22px; background-color: White; display: inline-block;">' \
                    'Video is <b>NOT</b> a DeepFake!</p>'
            text_yes = '<p style="font-family:Modern; color:Red; font-size: 22px; background-color:White; display: inline-block;">' \
                    'Video <b>IS</b> a DeepFake!</p>'
            text_maybe = '<p style="font-family:Modern; color:Orange; font-size: 22px; background-color:White; display: inline-block;">' \
                    'Video may be a DeepFake!</p>'

        return text_not if prob < lth else text_yes if prob > hth else text_maybe


def get_width(image_shape, max_width=400):
    if image_shape[1] > max_width:
        return max_width
    else:
        return image_shape[1]


def read_frames(folder, skip=5, n_frames=60):

    frames = []
    frame_names = os.listdir(folder)[::skip][0:n_frames]
    if len(frame_names) < n_frames:
        frame_names = frame_names + [None] * (60 - len(frame_names))

    frame_shape = (3, 299, 299)
    for frame in frame_names:
        if frame:
            img = os.path.join(folder, frame)
            X = np.array(Image.open(img), dtype=np.float32)
            if np.max(X) > 1:
                X = X / 255

            frame_shape = X.shape
            frames.append(X)
        else:
            frames.append(np.zeros(frame_shape, dtype=np.float32))

    return frames

def make_collage(frames):

    frames = frames[0:20]
    collage = None
    for i in range(int(len(frames) / 4)):
        new_img = np.hstack(frames[i*4:i*4+4])
        if collage is None:
            collage = new_img
        else:
            collage = np.vstack([collage, new_img])

    return collage

if __name__ == "__main__":

    side_bg = "ai.png"
    side_bg_ext = "png"
    main_bg = "ai.png"
    main_bg_ext = "png"
    uploaded_folder = r"C:\Users\user\Desktop\ML\AI4Media\Code\uploaded_files"
    processed_folder = r'C:\Users\user\Desktop\ML\AI4Media\Code\processed'

    arch2model = {'CapsNet (FF++)': ['capsule', 'FF'],
                  'ResNet-50 (FF++)': ['resnet-50', 'FF'],
                  'XceptionNet (FF++)': ['Xception', 'FF'],
                  'CapsNet (CelebDF)': ['capsule', 'celebDF'],
                  'ResNet-50 (CelebDF)': ['resnet-50', 'celebDF'],
                  'XceptionNet (CelebDF)': ['Xception', 'celebDF'],
                  'CapsNet-LSTM (CelebDF)': ['capsule-LSTM', 'celebDF'],
                  'ResNet-50-LSTM (FF++)': ['resnet-50-LSTM', 'FF'],
                  'ResNet-50-LSTM (CelebDF)': ['resnet-50-LSTM', 'celebDF'],
                  'XceptionNet-LSTM (CelebDF)': ['Xception-LSTM', 'celebDF'],
                  'XceptionNet-LSTM (FF++)': ['Xception-LSTM', 'FF'],

                  }

    transf_imagenet = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        # Resnet and VGG19 expects to have data normalized this way (because pretrained)
    ])

    transf_xception = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    st.markdown(
        f"""
        <style>
        .reportview-container {{
            background: url(data:image/{main_bg_ext};base64,{base64.b64encode(open(main_bg, "rb").read()).decode()});
            background-position: center;
           background-size: 100%;
    background-repeat: no-repeat;
    
        }}
       .sidebar .sidebar-content {{
            background: url(data:image/{side_bg_ext};base64,{base64.b64encode(open(side_bg, "rb").read()).decode()});
            background-position: center;
            background-size: 100%;
    background-repeat: no-repeat;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

    original_title = '<p style="font-family:Modern; color: #9EDBEB; font-size: 65px; text-align:center;"><b>DeepFake Detection</b></p>'
    st.markdown(original_title, unsafe_allow_html=True)

    text1 = '<p style="font-family:Modern; color:#9EDBEB; font-size: 20px;">' \
            'This is a simple web app that predicts if the uploaded image/video is a deepfake.</p>'
    st.markdown(text1, unsafe_allow_html=True)

    text2 = '<p style="font-family:Modern; color:#9EDBEB; font-size: 16px;">' \
            '<i>Please upload an image or a video</i></p>'
    st.markdown(text2, unsafe_allow_html=True)

    file = st.file_uploader("", type=["jpg", "png", "bmp", "mp4"])


    if file is None:
        text3 = '<p style="font-family:Modern; color:Red; font-size: 16px;">' \
                'File not uploaded</p>'
        st.markdown(text3, unsafe_allow_html=True)

    # If photo, run phtoto networks
    elif file.name.endswith('.jpg') or file.name.endswith('.png') or file.name.endswith('.bmp'):

        preprocess = st.checkbox("Preprocess File", value=True)

        # Saving image
        file_name = file.name.split('.')[0]
        uploaded_image = Image.open(file)
        uploaded_image.save(os.path.join(uploaded_folder, file.name))

        #Adding select bar
        text2 = '<p style="font-family:Modern; color:#9EDBEB; font-size: 16px;">' \
                '<i>Please select model(s)</i></p>'
        st.markdown(text2, unsafe_allow_html=True)
        dropdown_values = st.multiselect('', ['CapsNet (FF++)', 'ResNet-50 (FF++)', 'XceptionNet (FF++)',
                                              'CapsNet (CelebDF)', 'ResNet-50 (CelebDF)', 'XceptionNet (CelebDF)'])

        text3 = '<p style="font-family:Modern; color:#9EDBEB; font-size: 16px;">' \
                'Uploaded image:</p>'
        st.markdown(text3, unsafe_allow_html=True)
        st.image(uploaded_image, width=get_width(uploaded_image.size))


        # If architectures are selected
        if dropdown_values:
            button = st.button('Run')

            if button:
                # Starting preprocessing using OpenFace2
                if preprocess:
                    try:
                        os.mkdir(os.path.join(processed_folder, file_name))
                    except:
                        pass
                    command = r"C:\Users\user\Desktop\ML\AI4Media\Code\OpenFace20\OpenFace_2.2.0_win_x64\OpenFace_2.2.0_win_x64\FaceLandmarkImg.exe -f " \
                              + os.path.join(uploaded_folder, file.name) \
                              + " -out_dir " + os.path.join(processed_folder, file_name) \
                              + " -simsize 299 -format_vis_image png"

                    os.system(command)

                    aligned_image = Image.open(os.path.join(os.path.join(processed_folder, file_name, file_name + '_aligned'),
                                                            os.listdir(os.path.join(processed_folder, file_name, file_name + '_aligned'))[0]))

                    text3 = '<p style="font-family:Modern; color:#9EDBEB; font-size: 16px;">' \
                            'Preprocessed image:</p>'
                    st.markdown(text3, unsafe_allow_html=True)
                    st.image(aligned_image)

                else:
                    aligned_image = uploaded_image


                # Running ML architectures for images
                runner = ML_Runner()
                for architecture in dropdown_values:

                    text3 = '<p style="font-family:Modern; color:#9EDBEB; font-size: 20px;">' + architecture + '</p>'
                    st.markdown(text3, unsafe_allow_html=True)

                    if 'CapsNet' in architecture:

                        prob = runner.run_capsulenet(aligned_image, model_type=arch2model[architecture][0],
                                                     dataset=arch2model[architecture][1], transform=transf_imagenet,
                                                     multiple_frames=False)
                        if 'FF' in arch2model[architecture][1]:
                            text = runner.isdeepfake(float(prob.flatten()[0]), lth=0.4, hth=0.6)
                        else:
                            text = runner.isdeepfake(float(prob.flatten()[0]))
                        st.markdown(text, unsafe_allow_html=True)
                        prob_text = '<p style="font-family:Modern; color:Black; font-size: 18px; background-color:White; display: inline-block;">DeepFake Probability: ' +\
                                    str("{:.3f}".format(prob.flatten()[0])) + ' </p>'
                        st.markdown(prob_text, unsafe_allow_html=True)

                    else:
                        if 'Xception' in architecture:
                            transf = transf_xception
                        else:
                            transf = transf_imagenet
                        prob = runner.run_cnn(aligned_image, model_type=arch2model[architecture][0], dataset=arch2model[architecture][1], transform=transf)
                        text = runner.isdeepfake(float(prob.flatten()[0]))
                        st.markdown(text, unsafe_allow_html=True)
                        prob_text = '<p style="font-family:Modern; color:Black; font-size: 18px; background-color:White; display: inline-block;">DeepFake Probability: ' +\
                                    str("{:.3f}".format(prob.flatten()[0])) + ' </p>'
                        st.markdown(prob_text, unsafe_allow_html=True)

    elif file.name.endswith('.mp4'):

        file_name = file.name.split('.')[0]
        with open(os.path.join(uploaded_folder, file.name), "wb") as f:
            f.write(file.getbuffer())

        #Adding select bar
        text2 = '<p style="font-family:Modern; color:#9EDBEB; font-size: 16px;">' \
                '<i>Please select model(s)</i></p>'
        st.markdown(text2, unsafe_allow_html=True)
        dropdown_values = st.multiselect('', ['CapsNet (FF++)', 'ResNet-50 (FF++)', 'XceptionNet (FF++)',
                                              'CapsNet (CelebDF)', 'ResNet-50 (CelebDF)', 'XceptionNet (CelebDF)',
                                              'CapsNet-LSTM (CelebDF)', 'ResNet-50-LSTM (FF++)', 'ResNet-50-LSTM (CelebDF)',
                                              'XceptionNet-LSTM (CelebDF)', 'XceptionNet-LSTM (FF++)'])

        if dropdown_values:
            button = st.button('Run')

            if button:

                # Starting preprocessing using OpenFace2
                try:
                    os.mkdir(os.path.join(processed_folder, file_name))
                except:
                    pass
                command = r"C:\Users\user\Desktop\ML\AI4Media\Code\OpenFace20\OpenFace_2.2.0_win_x64\OpenFace_2.2.0_win_x64\FeatureExtraction.exe -f " \
                          + os.path.join(uploaded_folder, file.name) \
                          + " -out_dir " + os.path.join(processed_folder, file_name) \
                          + " -simsize 299 -format_aligned png"

                os.system(command)

                frames = read_frames(os.path.join(processed_folder, file_name, file_name + '_aligned'))
                collage = make_collage(frames)
                text3 = '<p style="font-family:Modern; color:#9EDBEB; font-size: 16px;">' \
                        'Preprocessed frames:</p>'
                st.markdown(text3, unsafe_allow_html=True)
                st.image(collage, use_column_width=True)

                # Running ML architectures for video
                runner = ML_Runner()
                for architecture in dropdown_values:

                    text3 = '<p style="font-family:Modern; color:#9EDBEB; font-size: 20px;">' + architecture + '</p>'
                    st.markdown(text3, unsafe_allow_html=True)

                    if 'LSTM' in architecture:

                        if 'Xception' in architecture:
                            transf = transf_xception
                        else:
                            transf = transf_imagenet
                        probs = runner.run_cnnlstm(frames, model_type=arch2model[architecture][0],
                                                     dataset=arch2model[architecture][1], transform=transf)
                        prob = probs[0]
                        text = runner.isdeepfake(float(prob.flatten()[0]), type='video')
                        st.markdown(text, unsafe_allow_html=True)
                        prob_text = '<p style="font-family:Modern; color:Black; font-size: 18px; background-color:White; display: inline-block;">DeepFake Probability: ' +\
                                    str("{:.3f}".format(prob.flatten()[0])) + ' </p>'
                        st.markdown(prob_text, unsafe_allow_html=True)


                    elif 'CapsNet' in architecture:

                        probs = runner.run_capsulenet(frames, model_type=arch2model[architecture][0],
                                                     dataset=arch2model[architecture][1], transform=transf_imagenet,
                                                     multiple_frames=True)

                        if 'FF' in arch2model[architecture][1]:
                            prob = torch.mean(probs[:, 0])
                            text = runner.isdeepfake(float(prob.flatten()[0]), lth=0.4, hth=0.6, type='video')
                        else:
                            prob = torch.mean(probs)
                            text = runner.isdeepfake(float(prob.flatten()[0]), type='video')
                        st.markdown(text, unsafe_allow_html=True)
                        prob_text = '<p style="font-family:Modern; color:Black; font-size: 18px; background-color:White; display: inline-block;">DeepFake Probability: ' +\
                                    str("{:.3f}".format(prob.flatten()[0])) + ' </p>'
                        st.markdown(prob_text, unsafe_allow_html=True)

                    else:
                        if 'Xception' in architecture:
                            transf = transf_xception
                        else:
                            transf = transf_imagenet
                        probs = runner.run_cnn(frames, model_type=arch2model[architecture][0],
                                               dataset=arch2model[architecture][1], transform=transf,
                                               multiple_frames=True)
                        prob = torch.mean(probs)
                        text = runner.isdeepfake(float(prob.flatten()[0]), type='video')
                        st.markdown(text, unsafe_allow_html=True)
                        prob_text = '<p style="font-family:Modern; color:Black; font-size: 18px; background-color:White; display: inline-block;">DeepFake Probability: ' +\
                                    str("{:.3f}".format(prob.flatten()[0])) + ' </p>'
                        st.markdown(prob_text, unsafe_allow_html=True)