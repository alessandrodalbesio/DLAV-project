import torch
from .base_dataset import BaseDataset
from torch_geometric.data import HeteroData
from motionnet.models.qcnet.utils import wrap_angle
import numpy as np

class QCNetDataset(BaseDataset):
    def __init__(self, config=None, is_validation=False, **kwargs):
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
        elem_converted['dataset_name'] = data['dataset_name']
            
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
        if self.config['store_data_in_memory']:
            for i in range(len(self.data_loaded_memory)):
                if i % 100 == 0:
                    print(f"Converting data {i+1} of {len(self.data_loaded_memory)}")
                self.data_loaded_memory[i][0] = HeteroData(self.prepare_data(self.data_loaded_memory[i][0]))

    def __getitem__(self, idx):
        # Get the item from super
        item = super().__getitem__(idx)[0]

        # Return the item
        if self.config['store_data_in_memory']:
            return item
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
        agent_type = torch.tensor(elem['obj_trajs'][:,:,6:11].argmax(axis=2)[0], dtype=torch.int)
        
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
        for i in range(num_agents):
            predict_mask[i, past_len:][not elem['obj_trajs_mask'][i,-1]] = False

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
        points_reshaped = points.reshape(-1,points.shape[-1])

        # Define the polygon - point mapping
        k, n = points.shape[0], points.shape[1]
        points_pol_indexes = np.repeat(np.arange(k), n)

        # Filter everything
        points_filtered = points_reshaped[mask_reshaped,:]
        points_pol_indexes_filtered = points_pol_indexes[mask_reshaped]
        points_indexes = np.arange(points_filtered.shape[0])

        # Define the empty dictionary
        map_data['map_point'] = {}
        map_data[('map_point','to','map_polygon')] = {}

        # Get the properties of the map points
        map_data['map_point']['position'] = torch.tensor(points_filtered[:,0:2], dtype=torch.float)
        map_data['map_point']['orientation'] = torch.tensor(np.arctan2(points_filtered[:,4],points_filtered[:,3]), dtype=torch.float)
        map_data['map_point']['height'] = torch.tensor(points_filtered[:,2], dtype=torch.float)
        map_data['map_point']['type'] = torch.tensor(points_filtered[:,6], dtype=torch.int)
        map_data['map_point']['num_nodes'] = points_filtered.shape[0]
        map_data[('map_point','to','map_polygon')]['edge_index'] = torch.tensor(np.array([points_indexes, points_pol_indexes_filtered]), dtype=torch.long)

        # Get the polygon data ("map_polygon")
        map_data['map_polygon'] = {}
        map_data[('map_polygon', 'to', 'map_polygon')] = {}

        # Get from points_pol_indexes_filtered the first index of each polygon
        first_indexes = np.unique(points_pol_indexes_filtered, return_index=True)[1]
        map_data['map_polygon']['position'] = torch.tensor(points_filtered[first_indexes,0:2], dtype=torch.float)
        map_data['map_polygon']['orientation'] = torch.tensor(np.arctan2(points_filtered[first_indexes,4],points_filtered[first_indexes,3]), dtype=torch.float)
        map_data['map_polygon']['height'] = torch.tensor(points_filtered[first_indexes,2], dtype=torch.float)
        map_data['map_polygon']['type'] = torch.tensor(points_filtered[first_indexes,6], dtype=torch.int)
        map_data['map_polygon']['num_nodes'] = first_indexes.shape[0]
        
        # Define the number of polygons
        N = first_indexes.shape[0]
        array = np.array([(i, j) for i in range(0, N) for j in range(0, N) if i != j])
        map_data[('map_polygon', 'to', 'map_polygon')]['edge_index'] = torch.tensor(array.T, dtype=torch.long)

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
        num_future_steps = self.config['num_future_steps']
        agent['target'] = origin.new_zeros(agent['num_nodes'], num_future_steps, 4)
        agent['target'][..., :2] = torch.bmm(agent['position'][:, self.config.num_historical_steps:, :2] - origin[:, :2].unsqueeze(1), rot_mat)
        if agent['position'].size(2) == 3:
            agent['target'][..., 2] = (agent['position'][:, self.config.num_historical_steps:, 2] - origin[:, 2].unsqueeze(-1))
        agent['target'][..., 3] = wrap_angle(agent['heading'][:, self.config.num_historical_steps:].squeeze() - theta.unsqueeze(-1))
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
