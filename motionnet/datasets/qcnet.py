import torch
from .base_dataset import BaseDataset
from torch_geometric.data import HeteroData

import numpy as np

class QCNetDataset(BaseDataset):
    def __init__(self, config=None, is_validation=False):
        super().__init__(config, is_validation)
        self.config = config

    def collate_fn(self, data):
        # List containing the data of the modified batch
        converted_batch = []

        # Iterate over the batch
        for batch in data:
            # Get the first element of the batch (list -> dict)
            batch = batch[0]

            # Define the dictionary that will contain the converted data
            elem_converted = dict()
            
            # Define the scenario_id
            elem_converted['scenario_id'] = batch['scenario_id']
            
            # Get the city name (use dataset name as city name for now)
            elem_converted['city'] = batch['dataset_name']
            
            # Define the agent data
            elem_converted['agent'] = self._convert_agent_data(batch)

            # Define the map data
            elem_converted.update(self._convert_map_data(batch))

            # Append the converted data to the converted batch
            converted_batch.append(elem_converted)
            
        # Return the converted batch
        return elem_converted

    def _convert_agent_data(self,elem):

        # Define the number of valid agents
        num_agents = self.config.method['max_num_agents']

        # Define the index to predict
        av_idx = elem['track_index_to_predict']

        # Define the agent data
        past_len = self.config['past_len']
        future_len = self.config['future_len']
        tot_time_len = past_len + future_len

        # Get the position, the heading and the velocity of the agents
        idx = torch.tensor(range(num_agents), dtype=torch.int) # Index of the agents
        agent_type = torch.zeros(num_agents, 5, dtype=torch.int) # Type of the agents
        position = torch.zeros(num_agents, tot_time_len, 2, dtype=torch.float) # Position of the agents
        heading = torch.zeros(num_agents, tot_time_len, 1, dtype=torch.float) # Heading of the agents
        velocity = torch.zeros(num_agents, tot_time_len, 2, dtype=torch.float) # Velocity of the agents
        valid_mask = torch.zeros(num_agents, tot_time_len, dtype=torch.bool) # Mask of the valid agents
        predict_mask = torch.zeros(num_agents, tot_time_len, dtype=torch.bool) # Mask of the agents to predict

        for i in range(num_agents):
            # Set the agent type
            agent_type[i, :] = torch.tensor(elem['obj_trajs'][i,0,6:11], dtype=torch.int)

            # Set the past position, velocity and heading
            position[i, :past_len, :] = torch.tensor(elem['obj_trajs'][i,:,:2])
            velocity[i, :past_len, :] = torch.tensor(elem['obj_trajs'][i,:,35:37])
            heading_angles = np.arctan2(elem['obj_trajs'][i,:,34], elem['obj_trajs'][i,:,33])[:, np.newaxis]
            heading[i, :past_len, :] = torch.tensor(heading_angles)

            # Set the future position, velocity and heading
            position[i, past_len:, :] = torch.tensor(elem['obj_trajs_future_state'][i, :, :2])
            velocity[i, past_len:, :] = torch.tensor(elem['obj_trajs_future_state'][i, :, 2:])
            heading_angles = np.arctan2(elem['obj_trajs_future_state'][i,:, 3], elem['obj_trajs_future_state'][i,:, 2])[:, np.newaxis]
            heading[i, past_len:, :] = torch.tensor(heading_angles)

            # Define all the validity masks
            valid_mask[i, :past_len] = torch.tensor(elem['obj_trajs_mask'][i])
            valid_mask[i, past_len:] = torch.tensor(elem['obj_trajs_future_mask'][i])
            predict_mask[i, :past_len] = torch.tensor([False]*past_len)
            predict_mask[i, past_len:] = torch.tensor([True]*future_len)
            if not elem['obj_trajs_mask'][i,-1]:
                predict_mask[i, :] = torch.tensor([False]*tot_time_len)

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

        # Get the valid points
        map_data['map_point'] = {'position': torch.tensor([]), 'orientation': torch.tensor([]), 'type': torch.tensor([]), 'height': torch.tensor([]), 'magnitude': torch.tensor([])}
        map_data[('map_point','to','map_polygon')] = {'edge_index': torch.tensor([], dtype=torch.long)}
        counter = 0
        counter_bis = 0

        # Create the map data: "map_point" and ("map_point", "to", "map_polygon")
        for i in range(elem['map_polylines'].shape[0]):
            valid_points = elem['map_polylines'][i,mask[i],:]
            if len(valid_points) > 0:                
                for j in range(len(valid_points)):
                    map_data['map_point']['position'] = torch.cat((map_data['map_point']['position'], torch.tensor(valid_points[j][0:2], dtype=torch.float)))
                    map_data['map_point']['orientation'] = torch.cat((map_data['map_point']['orientation'], torch.tensor([np.arctan2(valid_points[j][4],valid_points[j][3])], dtype=torch.float)))
                    map_data['map_point']['type'] = torch.cat((map_data['map_point']['type'], torch.tensor([int(valid_points[j][6])], dtype=torch.int)))
                    map_data['map_point']['height'] = torch.cat((map_data['map_point']['height'], torch.tensor([valid_points[j][2]], dtype=torch.float)))
                    magnitude = np.linalg.norm(valid_points[j][0:2]-valid_points[0][0:2])
                    map_data['map_point']['magnitude'] = torch.cat((map_data['map_point']['magnitude'], torch.tensor([magnitude], dtype=torch.float)))
                    # [('map_point','to','map_polygon')]['edge_index'] has first element equal to point index and second to polygon index
                    map_data[('map_point','to','map_polygon')]['edge_index'] = torch.cat((map_data[('map_point','to','map_polygon')]['edge_index'], torch.tensor([counter, counter_bis], dtype=torch.long)))
                    counter += 1
                counter_bis += 1
        map_data['map_point']['num_nodes'] = counter
        map_data['map_point']['position'] = map_data['map_point']['position'].reshape(-1,2)
        map_data[('map_point','to','map_polygon')]['edge_index'] = map_data[('map_point','to','map_polygon')]['edge_index'].reshape(-1,2).transpose(0,1)

        # Get the polygon data ("map_polygon")
        counter = 0
        map_data['map_polygon'] = {'position': torch.tensor([]), 'orientation': torch.tensor([]), 'type': torch.tensor([]), 'height': torch.tensor([])}
        for i in range(elem['map_polylines'].shape[0]):
            if mask[i][0]:
                counter += 1  
                valid_point = elem['map_polylines'][i,0,:]
                map_data['map_polygon']['position'] = torch.cat((map_data['map_polygon']['position'], torch.tensor(valid_point[0:2], dtype=torch.float)))
                map_data['map_polygon']['orientation'] = torch.cat((map_data['map_polygon']['orientation'], torch.tensor([np.arctan2(valid_point[4],valid_point[3])], dtype=torch.float)))
                map_data['map_polygon']['type'] = torch.cat((map_data['map_polygon']['type'], torch.tensor([valid_point[6]], dtype=torch.float)))
                map_data['map_polygon']['height'] = torch.cat((map_data['map_polygon']['height'], torch.tensor([valid_point[2]], dtype=torch.float)))
        map_data['map_polygon']['position'] = map_data['map_polygon']['position'].reshape(-1,2)

        map_data['map_polygon']['num_nodes'] = counter
        
        return map_data