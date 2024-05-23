import torch
from .base_dataset import BaseDataset
from torch_geometric.data import HeteroData
from motionnet.models.qcnet.utils import wrap_angle
import numpy as np

class QCNetDataset(BaseDataset):
    def __init__(self, config=None, is_validation=False):
        super().__init__(config, is_validation)
        self.config = config
        self.num_historical_steps = config['past_len']
        self.num_future_steps = config['future_len']

    def prepare_data(self,data):     
        # Define the dictionary that will contain the converted data
        elem_converted = dict()
            
        # Define the scenario_id
        elem_converted['scenario_id'] = data['scenario_id']
            
        # Get the city name (use dataset name as city name for now)
        elem_converted['city'] = data['dataset_name']
            
        # Define the agent data
        elem_converted['agent'] = self._convert_agent_data(data)

        # Add the target information to the elem_converted
        elem_converted['agent']['target'] = self._target_definition(elem_converted['agent'])

        # Add the ground truth trajectory information to the elem_converted (needed for evaluation)
        elem_converted.update(self._append_gt_data(data))

        # Define the map data
        elem_converted.update(self._convert_map_data(data))

        # Append the converted data to the converted batch
        return elem_converted

    def load_data(self):
        super().load_data()
        for i in range(len(self.data_loaded_memory)):
            self.data_loaded_memory[i] = self.prepare_data(self.data_loaded_memory[i])

    def __getitem__(self, index):
        # Get the item
        item = super().__getitem__(index)
        
        # Process the data
        if self.config['store_data_in_memory']:
            return HeteroData(self.data_loaded_memory[index])
        else:
            return HeteroData(self.prepare_data(item))

    def _convert_agent_data(self,elem):
        # Define the number of valid agents
        num_agents = self.config.method['max_num_agents']

        # Define the index to predict
        av_idx = elem['track_index_to_predict']

        # Define the agent data
        past_len = self.config['past_len']
        future_len = self.config['future_len']
        tot_time_len = past_len + future_len

        # Create the arrays to store the data
        idx = torch.tensor(range(num_agents), dtype=torch.int) # Index of the agents

        # Fill in the agent type
        agent_type = torch.zeros(num_agents, dtype=torch.int) # Type of the agents
        agent_type = torch.tensor(elem['obj_trajs'][:,0,6:11].argmax(axis=1), dtype=torch.int)
        
        # Fill in the position
        position = torch.zeros(num_agents, tot_time_len, 2, dtype=torch.float) # Position of the agents
        position[:, :past_len, :] = torch.tensor(elem['obj_trajs'][:,:,0:2])
        position[:, past_len:, :] = torch.tensor(elem['obj_trajs_future_state'][:,:,0:2])

        # Fill in the velocity
        velocity = torch.zeros(num_agents, tot_time_len, 2, dtype=torch.float) # Velocity of the agents
        velocity[:, :past_len, :] = torch.tensor(elem['obj_trajs'][:,:,35:37])
        velocity[:, past_len:, :] = torch.tensor(elem['obj_trajs_future_state'][:,:,2:4])

        # Fill in the heading
        heading = torch.zeros(num_agents, tot_time_len, 1, dtype=torch.float) # Heading of the agents
        heading_angles = np.arctan2(elem['obj_trajs'][:,:,34], elem['obj_trajs'][:,:,33])[:, :, np.newaxis]
        heading[:, :past_len, :] = torch.tensor(heading_angles)
        heading_angles_futures = np.arctan2(elem['obj_trajs_future_state'][:,:,3], elem['obj_trajs_future_state'][:,:,2])[:, :, np.newaxis]
        heading[:, past_len:, :] = torch.tensor(heading_angles_futures)

        # Fill in the valid mask
        valid_mask = torch.zeros(num_agents, tot_time_len, dtype=torch.bool) # Mask of the valid agents
        valid_mask[:, :past_len] = torch.tensor(elem['obj_trajs_mask'])
        valid_mask[:, past_len:] = torch.tensor(elem['obj_trajs_future_mask'])

        # Fill in the predict mask
        predict_mask = torch.zeros(num_agents, tot_time_len, dtype=torch.bool) # Mask of the agents to predict
        predict_mask[:, :past_len] = torch.tensor([False]*past_len)
        predict_mask[:, past_len:] = torch.tensor([True]*future_len)
        predict_mask[:, past_len:][elem['obj_trajs_mask'][:,-1]] = False

        # Return the prepared data
        return {
            'num_nodes': num_agents,
            'av_index': av_idx,
            'valid_mask': valid_mask,
            'predict_mask': predict_mask,
            'id': idx,
            'type': agent_type,
            'position': position,
            'heading': heading,
            'velocity': velocity
        }


    def _convert_map_data(self,elem):
        # Define the map data
        map_data = dict()

        # Get the mask of the polylines
        mask = elem['map_polylines_mask']
        mask_reshaped = mask.reshape(-1)

        # Define a flattened mask
        points = elem['map_polylines']
        points_reshaped = points.reshape(-1,points.shape[2])

        # Define the polygon - point mapping
        k, n = points.shape[0], points.shape[1]
        points_pol_indeces = np.repeat(np.arange(k), n)

        # Filter everything
        points_filtered = points_reshaped[mask_reshaped,:]
        points_pol_indeces_filtered = points_pol_indeces[mask_reshaped]
        points_indeces = np.arange(points_filtered.shape[0])

        # Define the empty dictionary
        map_data['map_point'] = {}
        map_data[('map_point','to','map_polygon')] = {}

        # Get the properties of the map points
        map_data['map_point']['position'] = torch.tensor(points_filtered[:,0:2], dtype=torch.float)
        map_data['map_point']['orientation'] = torch.tensor(np.arctan2(points_filtered[:,4],points_filtered[:,3]), dtype=torch.float)
        map_data['map_point']['height'] = torch.tensor(points_filtered[:,2], dtype=torch.float)
        map_data['map_point']['type'] = torch.tensor(points_filtered[:,6], dtype=torch.int)
        map_data['map_point']['magnitude'] = torch.zeros(points_filtered.shape[0], dtype=torch.float)
        for num_pol in range(k):
            i = np.argwhere(points_pol_indeces_filtered == num_pol)
            if len(i) != 0:
                map_data['map_point']['magnitude'][i[0]] = torch.tensor(0, dtype=torch.float)
                if len(i) > 1:
                    map_data['map_point']['magnitude'][i[1:]] = torch.tensor(np.linalg.norm(points_filtered[i[1:],0:2]-points_filtered[i[:-1],0:2], axis=2), dtype=torch.float)
        map_data['map_point']['num_nodes'] = points_filtered.shape[0]
        map_data[('map_point','to','map_polygon')]['edge_index'] = torch.tensor(np.array([points_indeces, points_pol_indeces_filtered]), dtype=torch.long)

        # Get the polygon data ("map_polygon")
        map_data['map_polygon'] = {}
        map_data[('map_polygon', 'to', 'map_polygon')] = {}

        # Get from points_pol_indeces_filtered the first index of each polygon
        first_indeces = np.unique(points_pol_indeces_filtered, return_index=True)[1]
        map_data['map_polygon']['position'] = torch.tensor(points_filtered[first_indeces,0:2], dtype=torch.float)
        map_data['map_polygon']['orientation'] = torch.tensor(np.arctan2(points_filtered[first_indeces,4],points_filtered[first_indeces,3]), dtype=torch.float)
        map_data['map_polygon']['height'] = torch.tensor(points_filtered[first_indeces,2], dtype=torch.float)
        map_data['map_polygon']['type'] = torch.tensor(points_filtered[first_indeces,6], dtype=torch.int)
        map_data['map_polygon']['num_nodes'] = first_indeces.shape[0]
        
        # Associate for each polygon index all the other polygon indeces
        #for i in range(first_indeces.shape[0]):
        #    indeces_other_pol = np.delete(np.arange(first_indeces.shape[0]),i)
        #    if len(indeces_other_pol) > 0:
        #        map_data[('map_polygon', 'to', 'map_polygon')]['edge_index'] = torch.cat((map_data[('map_polygon', 'to', 'map_polygon')]['edge_index'], torch.tensor([np.repeat(i,len(indeces_other_pol)), indeces_other_pol], dtype=torch.long), 1))
        
        # Return the map data
        return map_data

    def _target_definition(self,agent):
        origin = agent['position'][:, self.config.num_historical_steps - 1]
        theta = agent['heading'][:, self.config.num_historical_steps - 1].squeeze()
        cos, sin = theta.cos(), theta.sin()
        rot_mat = theta.new_zeros(agent['num_nodes'], 2, 2)
        rot_mat[:, 0, 0] = cos
        rot_mat[:, 0, 1] = -sin
        rot_mat[:, 1, 0] = sin
        rot_mat[:, 1, 1] = cos
        agent['target'] = origin.new_zeros(agent['num_nodes'], self.num_future_steps, 4)
        agent['target'][..., :2] = torch.bmm(agent['position'][:, self.config.num_historical_steps:, :2] -
                                                            origin[:, :2].unsqueeze(1), rot_mat)
        if agent['position'].size(2) == 3:
            agent['target'][..., 2] = (agent['position'][:, self.config.num_historical_steps:, 2] -
                                                    origin[:, 2].unsqueeze(-1))
            
        agent['target'][..., 3] = wrap_angle(agent['heading'][:, self.config.num_historical_steps:].squeeze() -
                                                                theta.unsqueeze(-1))
        return agent['target']
    
    def _append_gt_data(self,data):
        def _conversion(elem):
            try:
                return torch.from_numpy(np.stack(elem, axis=0))
            except:
                return elem
        
        # Converted element
        elem_converted = dict()

        # Append the ground truth information
        elem_converted['center_gt_trajs'] = _conversion(data['center_gt_trajs'])
        elem_converted['center_gt_trajs_mask'] = _conversion(data['center_gt_trajs_mask'])
        elem_converted['center_gt_final_valid_idx'] = _conversion(data['center_gt_final_valid_idx'])
        elem_converted['center_gt_trajs_src'] = _conversion(data['center_gt_trajs_src'])

        # Return the converted element
        return elem_converted