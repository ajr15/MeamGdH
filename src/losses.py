class AveragePairLoss:

    def __init__(self, energy_weight: float, force_weight: float, pair_loss_function: str):
        self.energy_weight = energy_weight
        self.force_weight = force_weight
        if pair_loss_function == "rmse":
            self.pair_loss_function = lambda true, pred: (true - pred) ** 2
        elif pair_loss_function == "mae":
            self.pair_loss_function = lambda true, pred: abs(true - pred)
        elif pair_loss_function == "mare":
            self.pair_loss_function = lambda true, pred: abs((true - pred) / true)
        else:
            raise ValueError("Unknown pair loss function {}. allowed options are rmse, mae, mare")

    def evaluate(self, true_energies: dict, true_forces: dict, pred_energies: dict, pred_forces: dict) -> float:
        # energy errors
        energy_err = 0
        for key in true_energies.keys():
            energy_err += self.pair_loss_function(true_energies[key], pred_energies[key])
        energy_err = energy_err / len(true_energies)
        # force errors
        force_err = 0
        counter = 0
        for key in true_forces.keys():
            for true_vec, pred_vec in zip(true_forces[key], pred_forces[key]):
                for tx, px in zip(true_vec, pred_vec):
                    counter += 1
                    force_err += self.pair_loss_function(tx, px)
        force_err = force_err / counter
        # returns weighted error
        return self.energy_weight * energy_err + self.force_weight * force_err
