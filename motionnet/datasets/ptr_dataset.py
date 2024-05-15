from .base_dataset import BaseDataset
import numpy as np

class PTRDataset(BaseDataset):
    def __init__(self, config=None, is_validation=False):
        super().__init__(config, is_validation)
        
        # Get the augmentation mode from the config
        self.aug_mode = config['aug_mode']
        assert self.aug_mode in ['none', 'on_circle', 'in_circle', 'hide_car'], 'Invalid augmentation mode: {}'.format(self.aug_mode)
        if self.aug_mode == 'none': # Skip the rest of the initialization if the augmentation mode is 'none'
            return 

        # Define the parameters based on the mode
        if self.aug_mode == 'on_circle' or self.aug_mode == 'in_circle':
            self._radius = config['radius']
        self._prob_car = config['prob_car']
        self._prob_aug = config['prob_aug']
        

    def _add_on_circle_noise(self):
        """ Generate a circle with radius self._radius and put the cars on it. """
        # Get the needed variables
        M,T = self.elem['obj_trajs_pos'].shape[:2]

        # Loop over each batch
        car_indices_modify = np.random.choice(M, int(M * self._prob_car), replace=False)
        for car_index in car_indices_modify:
            for point_index in range(T-1):
                # Randomly decide whether to modify the point
                if np.random.rand() > self._prob_aug:
                    continue
                # Generate a random angle and put the car on the circle
                angle = np.random.rand() * 2 * np.pi
                x = self.elem['obj_trajs_pos'][car_index, point_index, 0]
                y = self.elem['obj_trajs_pos'][car_index, point_index, 1]
                self.elem['obj_trajs_pos'][car_index, point_index, 0] = x + self._radius * np.cos(angle)
                self.elem['obj_trajs_pos'][car_index, point_index, 1] = y + self._radius * np.sin(angle)

    def _add_in_circle_noise(self):
        """ Generate a circle with radius self._radius and put the cars in it. """
        # Get the needed variables
        M,T = self.elem['obj_trajs_pos'].shape[:2]

        # Randomly choose the cars to modify
        car_indices_modify = np.random.choice(M, int(M * self._prob_car), replace=False)
        for car_index in car_indices_modify:
            for point_index in range(T-1):
                # Randomly decide whether to modify the point
                if np.random.rand() > self._prob_aug:
                    continue
                # Generate a random angle and radius and put the car in the circle
                angle = np.random.rand() * 2 * np.pi
                radius = np.random.rand() * self._radius
                x = self.elem['obj_trajs_pos'][car_index, point_index, 0]
                y = self.elem['obj_trajs_pos'][car_index, point_index, 1]
                self.elem['obj_trajs_pos'][car_index, point_index, 0] = x + radius * np.cos(angle)
                self.elem['obj_trajs_pos'][car_index, point_index, 1] = y + radius * np.sin(angle)
         
    def _hide_car(self):
        """Hide a random percentage of cars in the batch."""

        # Get the needed variables
        M,T = self.elem['obj_trajs_pos'].shape[:2]  # Get the batch size, number of cars, and number of points in the trajectory

        # Calculate the number of cars to hide
        num_cars_to_hide = int(M * self._prob_car)
        if num_cars_to_hide == 0:
            return

        # Get the object trajectories and masks from the batch dictionary
        car_indices_to_hide = np.random.choice(M, num_cars_to_hide, replace=False) # Randomly choose the cars to hide
        for car_index in car_indices_to_hide: # Loop over each car to hide
            for point_index in range(T-1):  # Loop over each point in the trajectory
                if np.random.rand() < self._prob_aug: # Randomly decide whether to hide the point
                    self.elem['obj_trajs_mask'][car_index, point_index] = 0 # Set the mask to zero --> this point won't be used

    def __getitem__(self, index):
        # Get the element from the parent class
        self.elem = super().__getitem__(index)[0]

        # Don't modify the element if it is a validation element
        if self.is_validation:
            return [self.elem]

        # Modify the element if it is a training element
        if self.aug_mode == 'on_circle':
            self._add_on_circle_noise()
        elif self.aug_mode == 'in_circle':
            self._add_in_circle_noise()
        elif self.aug_mode == 'hide_car':
            self._hide_car()

        # Return the modified element
        return [self.elem]

