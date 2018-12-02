
class NetworkData:
    def __init__(self):
        self.checkpoint_path: str = None
        self.model_path: str = None
        self.tensorboard_path: str = None

        self.num_features: int = None
        self.num_classes: int = None

        self.num_h1_units: int = None
        self.h1_activation = None
        self.h1_kernel_init = None
        self.h1_bias_int = None

        self.num_h2_units: int = None
        self.h2_activation = None
        self.h2_kernel_init = None
        self.h2_bias_int = None

        self.learning_rate: float = None
        self.adam_epsilon: float = None

        self.regularizer: float = None
