import os
import cv2
import random
import numpy as np

from tensorflow.python.client import device_lib
from keras.utils import to_categorical
from keras.preprocessing.image import apply_affine_transform
from keras.models import load_model, Sequential, Model
from keras.layers import average, TimeDistributed, GlobalAveragePooling1D, concatenate, Dense, CuDNNLSTM, Dropout
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras import optimizers



'''
These functions are passed as callbacks to the model to include them as evaluation metrics automatically
'''
from keras import backend as K

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

'''
Generator class used to produce the data to be fed into the network
'''
class FusedModelGenerator:
    def __init__(self, train_dir, test_dir, filter=None, num_of_snip=10, num_of_snip_spatial=1, opt_flow_len=10, batch_size=16, fileext_opticalflow=".jpg",
                 fileext_spatial=".jpg", spatial_input_name = 'spatial_input'):
        self.train_dir = train_dir
        self.test_dir = test_dir
        print(f"Generating from {self.train_dir} / {self.test_dir}")
        self.filter = filter
        self.n_snips = num_of_snip
        self.n_snips_spatial = num_of_snip
        self.opt_flow_len = opt_flow_len
        self.batch_size = batch_size
        self.fileext_opticalflow = fileext_opticalflow
        self.fileext_spatial = fileext_spatial
        self.spatial_input_name = spatial_input_name

        # File refers to a scan  (A class)
        self.classname_by_id = {i: file for i, file in enumerate(os.listdir(self.train_dir)) if
                                os.path.isdir(os.path.join(self.train_dir, file))}
        self.id_by_classname = {cls: i for i, cls in self.classname_by_id.items()}
        self.n_classes = len(self.classname_by_id)

        self.videopaths_opticalflow_train = self.get_videonames(self.train_dir)  # Includes class folder in path
        self.n_train = len(self.videopaths_opticalflow_train)
        self.videopaths_opticalflow_test = self.get_videonames(self.test_dir)
        self.n_test = len(self.videopaths_opticalflow_test)

        self.count = 0

  

    def _augment(self, image, row_axis = 0, col_axis = 1, channel_axis=2):
        # AFFINE
        # print('LINE 40 in AUGMENT FUNCTION: ', image.shape[col_axis])
        image = apply_affine_transform(image,
                                       theta=np.random.uniform(-5,5),
                                       tx=np.random.uniform(-0.1,0.1) * image.shape[col_axis],
                                       ty=np.random.uniform(-0.1,0.1) * image.shape[row_axis],
                                       shear=np.random.uniform(-5, 5),
                                       zx=np.random.uniform(0.9,1.1),
                                       zy=np.random.uniform(0.9,1.1),
                                       row_axis=row_axis,
                                       col_axis=col_axis,
                                       channel_axis=channel_axis,
                                       fill_mode='nearest',
                                       cval=0,
                                       order=1)
        return image

    def get_videonames(self, directory):
        # videonames is a list of paths to a video (single scan data point)
        videonames = []
        for classid, classdir in self.classname_by_id.items():
            # Directory = Whole train folder
            # Classdir refers to a scan (A class)
            # File here refers to a video (A folder of the images/flow data) in a Scan (Classdir)
            videonames.extend([os.path.join(classdir, file) for file in os.listdir(os.path.join(directory, classdir)) if
                               os.path.isdir(os.path.join(directory, classdir, file))])
        if self.filter:
            n_videos = len(videonames)
            videonames = [video for video in videonames if self.filter in video]
            print(f'Filtering by {self.filter}; {n_videos} -> {len(videonames)}')

        return videonames

    def generate_stacks(self, train_or_test):
        while True:

            if train_or_test == "train":
                data_dir = self.train_dir
                videonames = self.videopaths_opticalflow_train
                random.shuffle(videonames) # Only shuffle training samples
                augment = True
            elif train_or_test == 'test':
                data_dir = self.test_dir
                videonames = self.videopaths_opticalflow_test
                augment = False
            else:
                raise ValueError("train_or_test must be 'train' or 'test'")

            n_batches = int(len(videonames) / self.batch_size)
            for i in range(n_batches):
                x, y = {'temporal_input': [], self.spatial_input_name: []}, []

                videonames_batch = videonames[i * self.batch_size:(i + 1) * self.batch_size]
                for _ in range(self.batch_size):
                    videoname = videonames_batch.pop(0)
                    videopath = os.path.join(data_dir, videoname)

                    seires_optical_stacks = self.get_series_of_optical_flow_stacks(videopath, augment)

                    series_spatial_images = self.get_series_of_spatial_images(videopath, augment)

                    onehot_class = to_categorical(self.id_by_classname[os.path.basename(os.path.dirname(videopath))], self.n_classes)

                    x['temporal_input'].append(seires_optical_stacks)
                    x[self.spatial_input_name].append(series_spatial_images)
                    y.append(onehot_class)

                try:
                    x['temporal_input'] = np.stack(x['temporal_input'])
                    x[self.spatial_input_name] = np.stack(x[self.spatial_input_name])
                    y = np.stack(y)
                except ValueError:
                    'Bigman we get an error here'
                yield x, y

    def get_series_of_optical_flow_stacks(self, directory, augment):
        series_optical_flow_stacks = []

        # starting_frame = 0
        fc = len([entry for entry in os.listdir(directory) if os.path.isfile(os.path.join(directory, entry))])/3
        starting_frame = np.random.randint(fc-(self.opt_flow_len*self.n_snips))
        self.count = 0

        #N_snips == Chunks per video
        for i in range(self.n_snips):

            optical_flow_stack = []
            _from = starting_frame + (self.opt_flow_len * i)
            _to = starting_frame + (self.opt_flow_len * (i + 1))

            if i == 0: # Make sure frame 0 is skipped as no flow info; this means iter0 is 1-10, and iter1 10-19
                _from += 1
                _to += 1
            selected_frames = range(_from, _to)
            NoneCheck = False
            # print(f"Snip {i}: frames: {selected_frames}")
            for i_frame in selected_frames:
                filename_x = os.path.join(directory, f"flow_x_{i_frame:05d}{self.fileext_opticalflow}")
                img_x = cv2.imread(filename_x, 0)

                try:
                    img_x = img_x / 255.
                except TypeError:
                    NoneCheck = True
                    print(f"Failed to load {i_frame} from {directory} from frames {selected_frames}")


                filename_y = os.path.join(directory, f"flow_y_{i_frame:05d}{self.fileext_opticalflow}")
                img_y = cv2.imread(filename_y, 0)

                # noinspection PyUnresolvedReferences,PyProtectedMember
                try:
                    img_y = np.swapaxes(img_y, 0, 1)
                    img_y = img_y / 255.
                except np.AxisError:
                    print(f"Axis error when loading {i_frame} from {directory}: img_y: {img_y}\nimg_x was {img_x}")

                if not NoneCheck:
                    optical_flow_stack.append(img_x)
                    optical_flow_stack.append(img_y)
                    self.count +=1
                else:
                    break

            if NoneCheck:
                break

            optical_flow_stack = np.array(optical_flow_stack)
            optical_flow_stack = np.swapaxes(optical_flow_stack, 0, 1)
            optical_flow_stack = np.swapaxes(optical_flow_stack, 1, 2)

            if augment:
                optical_flow_stack = self._augment(optical_flow_stack)
            series_optical_flow_stacks.append(optical_flow_stack)
        #   series_optical_flow_stacks shape  = [chunks per vid (n_snips), 224, 224, 2*(flow length)]
        r = np.stack(series_optical_flow_stacks)
        return r

    def get_series_of_spatial_images(self, dir, augment):
        series_spatial_images = []

        if self.n_snips_spatial == self.n_snips:
            selected_frames = [i*self.opt_flow_len for i in range(self.n_snips)]
        else:
            selected_frames = np.linspace(0,self.opt_flow_len * self.n_snips - 1,20, dtype=np.uint)

        for i_frame in selected_frames:
            filename_spatial = os.path.join(dir, f"img_{i_frame+1:05d}{self.fileext_spatial}")
            img_spatial = cv2.imread(filename_spatial, 1)

            if augment:
                img_spatial = self._augment(img_spatial)

            img_spatial = img_spatial / 255.
            series_spatial_images.append(img_spatial)

        series_spatial_images = np.array(series_spatial_images)
        return series_spatial_images
'''
Neural Netowrk Architecture being trained on the FAST dataset
'''
class TwoStreamFused():
    def __init__(self, spatial_model_name, temporal_model_name,
                 train_dir, test_dir,
                 width_temporal, height_temporal,
                 width_spatial, height_spatial,
                 opt_flow_len,
                 chunks_per_video,
                 batch_size,
                 learning_rate,
                 decay,
                 spatial_input_name = 'spatial_input', pop_classifiers=True, filter=None, frozen_spatial=True, frozen_temporal=True, load_td_spatial_model=False, classifier_in_submodel=False, tensorboard_dir='./Models/ViewClassifier/tensorboard_logs/'):
        self.spatial_model_name = spatial_model_name
        self.temporal_model_name = temporal_model_name
        self.modelname = f"Fused_{os.path.basename(spatial_model_name)}_{os.path.basename(temporal_model_name)}"
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.pop_classifiers = pop_classifiers
        self.filter = filter
        self.width_temporal = width_temporal
        self.height_temporal = height_temporal
        self.width_spatial = width_spatial
        self.height_spatial = height_spatial
        self.opt_flow_len = opt_flow_len
        self.chunks_per_video = chunks_per_video
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        self.decay = decay
        self.opt = optimizers.Adam(learning_rate=self.learning_rate, decay=self.decay)

        self.frozen_spatial = frozen_spatial
        self.frozen_temporal = frozen_temporal
        self.load_td_spatial_model = load_td_spatial_model
        self.classifier_in_submodel=classifier_in_submodel
        self.tensorboard_dir = tensorboard_dir
        self.spatial_input_name = spatial_input_name
        self.model_dir = os.path.dirname(os.path.dirname(tensorboard_dir))

        log_dir = os.path.join(self.tensorboard_dir, self.modelname)
        self.tbCallBack = TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=True, write_images=True)
        self.checkpointCallBack = ModelCheckpoint(
            f"{os.path.join(self.model_dir, self.modelname)}" + "-e{epoch:02d}-acc{val_accuracy:.3f}.hdf5",
            monitor='val_accuracy',
            verbose=1, save_best_only=True, mode='max')

        self.generator = FusedModelGenerator(train_dir=self.train_dir,
                                             test_dir=self.test_dir,
                                             filter=self.filter,
                                             num_of_snip=self.chunks_per_video,
                                             opt_flow_len=self.opt_flow_len,
                                             batch_size=self.batch_size,
                                             fileext_opticalflow=".jpg",
                                             fileext_spatial='.jpg',
                                             spatial_input_name=spatial_input_name)

        self.temporal_model = self.load_temporal_model()
        self.spatial_model = self.load_spatial_model()
        self.model = self.load_model()
        self.train_generator = self.generator.generate_stacks('train')
        self.test_generator = self.generator.generate_stacks('test')

    def load_spatial_model(self):
        spatial_model = load_model(self.spatial_model_name)
        if self.classifier_in_submodel:
            spatial_model = spatial_model.layers[0]
        print(f"Loaded {self.spatial_model_name} for spatial stream")
        if self.pop_classifiers and not self.classifier_in_submodel:
            # print(f"Popping classifier from spatial model")
            spatial_model.layers.pop()
            # input shape (-1,-1,-1,3)
            # output shape (-1,2048)
            spatial_model = Model(inputs=spatial_model.layers[0].input, outputs=spatial_model.layers[-2].output)
            spatial_model.compile(loss='categorical_crossentropy', optimizer=self.opt, metrics=['accuracy', f1_m, precision_m, recall_m])
        if self.frozen_spatial:
            for layer in spatial_model.layers:
                layer.trainable = False
        if not self.load_td_spatial_model:
            model = Sequential(name='spatial')
            model.add(TimeDistributed((spatial_model),
                                      input_shape=(self.chunks_per_video,
                                                   self.width_spatial,
                                                   self.height_spatial,
                                                   3),
                                      name='spatial'))
            model.add(GlobalAveragePooling1D())
            return model
        else:
            return spatial_model

    def load_temporal_model(self):
        temporal_model = load_model(self.temporal_model_name)
        print(f"Loaded {self.temporal_model_name} for temporal stream")
        if self.pop_classifiers:
            # print(f"Popping classifier from temporal model:\n{.temporal_model.summary()}")
            temporal_model.layers.pop()
            temporal_model = Model(inputs=temporal_model.layers[0].input, outputs=temporal_model.layers[-1].output)
            temporal_model.compile(loss='categorical_crossentropy', optimizer=self.opt, metrics=['accuracy', f1_m, precision_m, recall_m])
        if self.frozen_temporal:
            for layer in temporal_model.layers:
                layer.trainable = False
        model = Sequential(name='temporal')

        model.add(TimeDistributed((temporal_model),
                                 input_shape=(self.chunks_per_video,
                                             self.width_temporal,
                                             self.height_temporal,
                                            self.opt_flow_len * 2),
                                name='temporal'))

        model.add(GlobalAveragePooling1D(name="temporal_global_average_pooling1d"))
        return model

    def load_model(self):
        if self.classifier_in_submodel:
            merge = concatenate([self.spatial_model.layers[-1].output, self.temporal_model.output])
            outputs = Dense(self.generator.n_classes, activation='softmax')(merge)
            model = Model([self.spatial_model.layers[0].input, self.temporal_model.input], outputs)

        else:
            merge = concatenate([self.spatial_model.output, self.temporal_model.output])
            outputs = Dense(self.generator.n_classes, activation='softmax')(merge)
            model = Model([self.spatial_model.input, self.temporal_model.input], outputs)

        model.compile(loss='categorical_crossentropy', optimizer=self.opt, metrics=['accuracy', f1_m, precision_m, recall_m])
        print(f"Final model summary:")
        model.summary()
        return model

    def train(self, epochs):
        if self.generator.test_dir:
            self.model.fit(self.train_generator,
                                     steps_per_epoch=self.generator.n_train // self.generator.batch_size,
                                     epochs=epochs,
                                     callbacks=[self.tbCallBack, self.checkpointCallBack],
                                     validation_data=self.test_generator,
                                     validation_steps=self.generator.n_test // self.generator.batch_size)
