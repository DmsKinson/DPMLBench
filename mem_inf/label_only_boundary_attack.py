from art.attacks.inference.membership_inference.label_only_boundary_distance import LabelOnlyDecisionBoundary
from art.utils import check_and_transform_label_format
import numpy as np
from typing import Optional


class LabelOnlyAttack(LabelOnlyDecisionBoundary):
    def infer(self, x: np.ndarray, y: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        """
        Infer membership of input `x` in estimator's training data.

        :param x: Input data.
        :param y: True labels for `x`.
        :param probabilities: a boolean indicating whether to return the predicted probabilities per class, or just
                                the predicted class

        :Keyword Arguments for HopSkipJump:
            * *norm*: Order of the norm. Possible values: "inf", np.inf or 2.
            * *max_iter*: Maximum number of iterations.
            * *max_eval*: Maximum number of evaluations for estimating gradient.
            * *init_eval*: Initial number of evaluations for estimating gradient.
            * *init_size*: Maximum number of trials for initial generation of adversarial examples.
            * *verbose*: Show progress bars.

        :return: An array holding the inferred membership status, 1 indicates a member and 0 indicates non-member,
                    or class probabilities.
        """
        from hsj import HopSkipJump

        if y is None:  # pragma: no cover
            raise ValueError("Argument `y` is None, but this attack requires true labels `y` to be provided.")

        if self.distance_threshold_tau is None:  # pragma: no cover
            raise ValueError(
                "No value for distance threshold `distance_threshold_tau` provided. Please set"
                "`distance_threshold_tau` or run method `calibrate_distance_threshold` on known training and test"
                "dataset."
            )

        if "probabilities" in kwargs.keys():
            probabilities = kwargs.get("probabilities")
            del kwargs["probabilities"]
        else:
            probabilities = False

        if "classifier" in kwargs:  # pragma: no cover
            raise ValueError("Keyword `classifier` in kwargs is not supported.")

        if "targeted" in kwargs:  # pragma: no cover
            raise ValueError("Keyword `targeted` in kwargs is not supported.")

        y = check_and_transform_label_format(y, self.estimator.nb_classes)

        hsj = HopSkipJump(classifier=self.estimator, targeted=False, **kwargs)
        x_adv = hsj.generate(x=x, y=y)

        distance = np.linalg.norm((x_adv - x).reshape((x.shape[0], -1)), ord=2, axis=1)

        y_pred = self.estimator.predict(x=x)

        distance[np.argmax(y_pred, axis=1) != np.argmax(y, axis=1)] = 0
        predicted_class = np.where(distance > self.distance_threshold_tau, 1, 0)
        if probabilities:
            prob_1 = np.zeros_like(distance)
            if self.threshold_bins:
                # bin accuracy is the probability of being a member
                for t_bin in self.threshold_bins:
                    prob_1[distance > t_bin[0]] = t_bin[1]
            else:
                # use sigmoid on distance from threshold
                dist_threshold = distance - self.distance_threshold_tau
                prob_1 = 1 / (1 + np.exp(-dist_threshold))
            prob_0 = np.ones_like(prob_1) - prob_1
            return np.stack((prob_0, prob_1), axis=1)
        return predicted_class

    def calibrate_distance_threshold(
        self, x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, y_test: np.ndarray, **kwargs
    ):
        """
        Calibrate distance threshold maximising the membership inference accuracy on `x_train` and `x_test`.

        | Paper link: https://arxiv.org/abs/2007.14321

        :param x_train: Training data.
        :param y_train: Labels of training data `x_train`.
        :param x_test: Test data.
        :param y_test: Labels of test data `x_test`.

        :Keyword Arguments for HopSkipJump:
            * *norm*: Order of the norm. Possible values: "inf", np.inf or 2.
            * *max_iter*: Maximum number of iterations.
            * *max_eval*: Maximum number of evaluations for estimating gradient.
            * *init_eval*: Initial number of evaluations for estimating gradient.
            * *init_size*: Maximum number of trials for initial generation of adversarial examples.
            * *verbose*: Show progress bars.
        """
        from hsj import HopSkipJump

        if "classifier" in kwargs:  # pragma: no cover
            raise ValueError("Keyword `classifier` in kwargs is not supported.")

        if "targeted" in kwargs:  # pragma: no cover
            raise ValueError("Keyword `targeted` in kwargs is not supported.")

        y_train = check_and_transform_label_format(y_train, self.estimator.nb_classes)
        y_test = check_and_transform_label_format(y_test, self.estimator.nb_classes)

        hsj = HopSkipJump(classifier=self.estimator, targeted=False, **kwargs)

        x_train_adv = hsj.generate(x=x_train, y=y_train)
        x_test_adv = hsj.generate(x=x_test, y=y_test)

        distance_train = np.linalg.norm((x_train_adv - x_train).reshape((x_train.shape[0], -1)), ord=2, axis=1)
        distance_test = np.linalg.norm((x_test_adv - x_test).reshape((x_test.shape[0], -1)), ord=2, axis=1)

        y_train_pred = self.estimator.predict(x=x_train)
        y_test_pred = self.estimator.predict(x=x_test)

        distance_train[np.argmax(y_train_pred, axis=1) != np.argmax(y_train, axis=1)] = 0
        distance_test[np.argmax(y_test_pred, axis=1) != np.argmax(y_test, axis=1)] = 0

        num_increments = 100
        tau_increment = np.amax([np.amax(distance_train), np.amax(distance_test)]) / num_increments

        acc_max = 0.0
        distance_threshold_tau = 0.0
        self.threshold_bins = []
        for i_tau in range(1, num_increments):
            # searching for threshold that yields the best accuracy in separating between members and non-members
            is_member_train = np.where(distance_train > i_tau * tau_increment, 1, 0)
            is_member_test = np.where(distance_test > i_tau * tau_increment, 1, 0)

            acc = (np.sum(is_member_train) + (is_member_test.shape[0] - np.sum(is_member_test))) / (
                is_member_train.shape[0] + is_member_test.shape[0]
            )
            new_threshold_tau = i_tau * tau_increment
            self.threshold_bins.append((new_threshold_tau, acc))
            if acc > acc_max:
                distance_threshold_tau = new_threshold_tau
                acc_max = acc

        self.distance_threshold_tau = distance_threshold_tau
