import sys

sys.path.append('./')
sys.path.append('../')
sys.path.append('../../')

from torch.utils.data import Dataset
from scipy.spatial.transform import Rotation as R


from surfaces.metric_resample import *
from surfaces.metric_resample_labels import *


class dataset_cortical_surfaces(Dataset):
    def __init__(self, 
                data_path,
                config,
                split,
                ):

        super().__init__()

        ################################################
        ##############       CONFIG       ##############
        ################################################

        self.task = config['data']['task']
        ico = config['mesh_resolution']['ico_mesh']
        sub_ico = config['mesh_resolution']['ico_grid']
        
        self.filedir =  data_path
        self.split = split
        self.configuration = config['data']['configuration']
        self.dataset = config['data']['dataset']
        self.path_to_workdir = config['data']['path_to_workdir']
        self.dataset = config['data']['dataset']
        self.augmentation = config['augmentation']['prob_augmentation']
        self.normalise = config['data']['normalise']
        self.clipping = config['data']['clipping']
        self.path_to_template = config['data']['path_to_template']
        self.warps_ico = config['augmentation']['warp_ico']
        self.nbr_vertices = config['sub_ico_{}'.format(sub_ico)]['num_vertices']
        self.nbr_patches = config['sub_ico_{}'.format(sub_ico)]['num_patches']

        if config['MODEL'] == 'sit' or config['MODEL']=='ms-sit':
            self.patching=True
            self.channels = config['data']['channels']
            self.num_channels = len(self.channels)
        else:
            raiseExceptions('model not implemented yet')

        ################################################
        ##############       LABELS       ##############
        ################################################
        self.data_info = pd.read_csv('{}/labels/{}/{}/half/{}.csv'.format(config['data']['path_to_workdir'], 
                                                                          self.dataset, self.task, split))

        self.filenames = self.data_info['ids']
        self.labels = self.data_info['labels']

        ###################################################
        ##############       NORMALISE       ##############
        ###################################################
        if self.normalise=='group-standardise':
        
            self.means = np.load('{}/labels/{}/cortical_metrics/{}/{}/{}/means.npy'.format(config['data']['path_to_workdir'],
                                                                                self.dataset,
                                                                                self.task,
                                                                                self.hemi,
                                                                                self.configuration))
            self.stds = np.load('{}/labels/{}/cortical_metrics/{}/{}/{}/stds.npy'.format(config['data']['path_to_workdir'],
                                                                                self.dataset,
                                                                                self.task,
                                                                                self.hemi,
                                                                                self.configuration))
        

        ########################################################################
        ##############       DATA AUGMENTATION & PROCESSING       ##############
        ########################################################################
        self.triangle_indices = pd.read_csv('{}/patch_extraction/triangle_indices_ico_{}_sub_ico_{}.csv'.format(config['data']['path_to_workdir'],ico,sub_ico))

        #config, augmentation
        if self.augmentation:
            self.rotation = config['augmentation']['prob_rotation']
            self.max_degree_rot = config['augmentation']['max_abs_deg_rotation']
            self.shuffle = config['augmentation']['prob_shuffle']
            self.warp = config['augmentation']['prob_warping']
            self.coord_ico6 = np.array(nb.load('{}/coordinates_ico_6_L.shape.gii'.format(self.path_to_template)).agg_data()).T        

        if config['mesh_resolution']['reorder']:
            #reorder patches 
            new_order_indices = np.load('{}/patch_extraction/reorder_patches/order_ico{}.npy'.format(config['data']['path_to_workdir'],sub_ico))
            d = {str(new_order_indices[i]):str(i) for i in range(len(self.triangle_indices.columns))}
            self.triangle_indices = self.triangle_indices[list([str(i) for i in new_order_indices])]
            self.triangle_indices = self.triangle_indices.rename(columns=d)
        
        self.masking = config['data']['masking']    
        if self.masking and self.dataset == 'dHCP' and self.configuration == 'template' : # for dHCP
            if split == 'train':
                print('Masking the cut: dHCP mask')
            self.mask = np.array(nb.load('{}/week-40_hemi-left_space-dhcpSym_dens-40k_desc-medialwallsymm_mask.shape.gii'.format(self.path_to_template)).agg_data())
        else:
            if split == 'train':
                print('Masking the cut: NO')
            
    def __len__(self):
        return len(self.filenames)

    def __getitem__(self,idx):
        ############
        # 1. load input metric
        # 2. select input channels
        # 3. mask input data (if masking)
        # 4. clip input data (if clipping)
        # 5. remask input data (if masking)
        # 6. normalise data (if normalise)
        # 7. apply augmentation
        # 8. get sequence of patches
        ############

        ### label
        #print(self.labels)
        label = self.labels.iloc[idx]

        ### hemisphere
        data = self.get_half_hemi(idx)
        if (self.augmentation and self.split =='train'):
            # do augmentation or not do augmentation
            if np.random.rand() > (1-self.augmentation):
                # chose which augmentation technique to use
                p = np.random.rand()
                if p < self.rotation:
                    #apply rotation
                    data = self.apply_rotation(data)

                elif self.rotation <= p < self.rotation+self.warp:
                    #apply warp
                    data = self.apply_non_linear_warp(data)

                elif self.rotation+self.war <= p < self.rotation+self.warp+self.shuffle:
                    #apply shuffle
                    data = self.apply_shuffle(data)
                    
        if self.patching:

            sequence = self.get_sequence(data)

            if self.augmentation and self.shuffle and self.split == 'train':
                bool_select_patch =np.random.rand(sequence.shape[1])<self.prob_shuffle
                copy_data = sequence[:,bool_select_patch,:]
                ind = np.where(bool_select_patch)[0]
                np.random.shuffle(ind)  
                sequence[:,ind,:]=copy_data
            return (torch.from_numpy(sequence).float(),torch.from_numpy(np.asarray(label,dtype=float)).float())

        else:
            return (torch.from_numpy(data).float(),torch.from_numpy(np.asarray(label,dtype=float)).float())

        return (torch.from_numpy(sequence).float(),torch.from_numpy(np.asarray(label,dtype=float)).float())

    
    def get_half_hemi(self,idx):

        #### 1. masking
        #### 2. normalising - only vertices that are not masked

        path = os.path.join(self.filedir,self.filenames.iloc[idx])
        data =  np.array(nb.load(path).agg_data())
        if len(data.shape)==1:
            data = np.expand_dims(data,0)
        data = data[self.channels,:]

        # load individual mask if native
        if self.masking and self.dataset=='dHCP' and self.configuration=='native':
            self.mask = np.array(nb.load(os.path.join(self.filedir, 'native_masks','{}.medialwall_mask.shape.gii'.format(self.filenames.iloc[idx].split('.')[0]))).agg_data())

        if self.masking:
            data = np.multiply(data,self.mask)
            if self.clipping:
                data = self.clipping_(data)
            data = np.multiply(data,self.mask) ## need a second masking to remove the artefacts after clipping
        else:
            if self.clipping:
                data = self.clipping_(data)

        data = self.normalise_(data)

        return data

    def clipping_(self,data):

        if self.dataset == 'dHCP':
            lower_bounds = np.array([0.0,-0.5, -0.05, -10.0])
            upper_bounds = np.array([2.2, 0.6,2.6, 10.0 ])
            for i,channel in enumerate(self.channels):
                data[i,:] = np.clip(data[i,:], lower_bounds[channel], upper_bounds[channel]
                )
       
        return data

    def normalise_(self,data):

        if self.masking:
            non_masked_vertices = self.mask>0
            if self.normalise=='group-standardise':
                data[:,non_masked_vertices] = (data[:,non_masked_vertices] - self.means[:,self.channels,:].reshape(self.num_channels,1))/self.stds[:,self.channels,:].reshape(self.num_channels,1)
            elif self.normalise=='sub-standardise':
                data[:,non_masked_vertices] = (data[:,non_masked_vertices] - data[:,non_masked_vertices].mean(axis=1).reshape(self.num_channels,1))/data[:,non_masked_vertices].std(axis=1).reshape(self.num_channels,1)
            elif self.normalise=='sub-normalise':
                data[:,non_masked_vertices] = (data[:,non_masked_vertices] - data[:,non_masked_vertices].min(axis=1).reshape(self.num_channels,1))/(data[:,non_masked_vertices].max(axis=1).reshape(self.num_channels,1)- data[:,non_masked_vertices].min(axis=1).reshape(self.num_channels,1))
        
        else:
            if self.normalise=='group-standardise':
                data= (data- self.means.reshape(self.num_channels,1))/self.stds.reshape(self.num_channels,1)
            elif self.normalise=='sub-standardise':
                data = (data - data.mean(axis=1).reshape(self.num_channels,1))/data.std(axis=1).reshape(self.num_channels,1)
            elif self.normalise=='normalise':
                data = (data- data.min(axis=1).reshape(self.num_channels,1))/(data.max(axis=1).reshape(self.num_channels,1)- data.min(axis=1).reshape(self.num_channels,1))
        return data
    
    ############ AUGMENTATION ############

    def get_sequence(self,data):

        sequence = np.zeros((self.num_channels, self.nbr_patches, self.nbr_vertices))
        for j in range(self.nbr_patches):
            indices_to_extract = self.triangle_indices[str(j)].to_numpy()
            sequence[:,j,:] = data[:,indices_to_extract]
        return sequence

    def apply_rotation(self,data):

        img = lat_lon_img_metrics('{}/surfaces/'.format(self.path_to_workdir),torch.Tensor(data.T).to('cpu'),device='cpu')

        rotation_angle = np.round(random.uniform(-self.max_degree_rot,self.max_degree_rot),2)
        axis = random.choice(['x','y','z'])

        r = R.from_euler(axis, rotation_angle, degrees=True)

        new_coord = np.asarray(r.apply(self.coord_ico6),dtype=np.float32)

        rotated_moving_img = bilinear_sphere_resample(torch.Tensor(new_coord),img, 100, 'cpu')

        return rotated_moving_img.numpy().T

    def apply_non_linear_warp(self,data):

        # chose one warps at random between 0 - 99

        id = np.random.randint(0,100)
        img = lat_lon_img_metrics('{}/surfaces/'.format(self.path_to_workdir),torch.Tensor(data.T).to('cpu'),device='cpu')
        warped_grid = nb.load('{}/warps/resample_ico6_ico_{}/ico_{}_{}.surf.gii'.format(self.path_to_template,self.warps_ico,self.warps_ico, id)).agg_data()
        warped_moving_img = bilinear_sphere_resample(torch.Tensor(warped_grid[0]), img, 100, 'cpu')
        return warped_moving_img.numpy().T
    
    ############ LOGGING ############

    def logging(self):
        
        if self.split == 'train':
                print('Using {} channels'.format(self.channels))
        
        if self.split == 'train':
            if self.normalise == 'sub-standardise':
                print('Normalisation: Subject-wise standardised')
            elif self.normalise == 'group-standardise':
                print('Normalisation: Group-wise standardised')
            elif self.normalise == 'normalise':
                print('Normalisation: Normalised')
            else:
                print('Normalisation: Not normalised') 

        print('')
        print('#'*30)
        print('######## Augmentation ########')
        print('#'*30)
        print('')
        if self.augmentation:
            print('Augmentation: ratio {}'.format(self.augmentation))
            if self.rotation:
                print('     - rotation with probability: {} and max abs degree {}'.format(self.rotation,self.max_degree_rot))
            else:
                print('     - rotations: no')
            if self.shuffle:
                print('     - shuffling with probability: {}'.format(self.shuffle))
            else:
                print('     - shuffling: no')
            if self.warp:
                print('     - non-linear warping with probability: {}'.format(self.warp))
            else:
                print('     - non-linear warping: no')
        else:
            print('Augmentation: NO')