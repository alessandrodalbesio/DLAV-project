from .base_dataset import BaseDataset

class PTRDataset(BaseDataset):

    def __init__(self, config=None, is_validation=False):
        super().__init__(config, is_validation)
        # Load the configuration
        self._hide_car_perc = config.method.hide_car_perc
        self._hide_car_prob = config.method.hide_car_prob
        self._noisy_car_perc = config.method.noisy_car_perc
        self._noisy_car_prob = config.method.noisy_car_prob
        
        print(f"Hide car perc: {self._hide_car_perc}")
        print(f"Hide car prob: {self._hide_car_prob}")
        print(f"Noisy car perc: {self._noisy_car_perc}")
        print(f"Noisy car prob: {self._noisy_car_prob}")

    def _hide_car(self,perc_car=0.5,prob_point=0.5):
        """
        TODO

        Args:
            perc_car (float): Percentage of the car to hide
            prob_point (float): Probability of hiding a point in the car trajectory
        """
        # Validate the input data
        assert perc_car >= 0 and perc_car <= 1
        assert prob_point >= 0 and prob_point <= 1

        # Do stuff
        pass

    def _noisy_car(self,perc_car=0.5,prob_point=0.5):
        """
        TODO

        Args:
            perc_car (float): Percentage of the car to hide
            prob_point (float): Probability of hiding a point in the car trajectory
        """

        def _on_circle(radius, center):
            # Move the center to the circle
            pass
        def _in_circle(radius, center):
            # Move the center inside the circle
            pass

        # Validate the input data
        assert perc_car >= 0 and perc_car <= 1
        assert prob_point >= 0 and prob_point <= 1

        # Do stuff
        pass

    def collate_fn(self, data_list):
        # Call the collate_fn of the base class
        batch_dict = super().collate_fn(data_list)

        # Optimize
        for i in range(32):
            input_dict = batch_dict['input_dict']
            obj_traj_mask = input_dict['obj_trajs_mask'][i,:,-1]
            obj_trajs_pos = input_dict['obj_trajs_pos'][i,:,-1]
            obj_trajs_last_pos = input_dict['obj_trajs_last_pos'][i,:]
            # If any of the obj_traj_mask is 0
            if False in obj_traj_mask:
                for j in range(len(obj_traj_mask)):
                    print(obj_traj_mask[j], obj_trajs_pos[j], obj_trajs_last_pos[j])
                breakpoint()

        # Do data augmentation on batch_dict
        if self.is_validation == False:
            # TODO: Prepare the data
            self._hide_car()
            self._noisy_car()

        # Return the augmented batch_dict
        return batch_dict
