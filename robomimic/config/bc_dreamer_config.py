"""
Config for BC algorithm.
"""

from robomimic.config.base_config import BaseConfig


class BCDreamerConfig(BaseConfig):
    ALGO_NAME = "bc_dreamer"

    def algo_config(self):
        """
        This function populates the `config.algo` attribute of the config, and is given to the 
        `Algo` subclass (see `algo/algo.py`) for each algorithm through the `algo_config` 
        argument to the constructor. Any parameter that an algorithm needs to determine its 
        training and test-time behavior should be populated here.
        """
        super(BCDreamerConfig, self).algo_config()

        # optimization parameters
        self.algo.optim_params.dynamics.learning_rate.initial = 1e-4      # world model rate
        self.algo.optim_params.dynamics.learning_rate.decay_factor = 0.1  # factor to decay LR by (if epoch schedule non-empty)
        self.algo.optim_params.dynamics.learning_rate.epoch_schedule = [] # epochs where LR decay occurs
        self.algo.optim_params.dynamics.regularization.L2 = 0.00          # L2 regularization strength

        self.algo.optim_params.policy.learning_rate.initial = 1e-4      # policy learning rate
        self.algo.optim_params.policy.learning_rate.decay_factor = 0.1  # factor to decay LR by (if epoch schedule non-empty)
        self.algo.optim_params.policy.learning_rate.epoch_schedule = [] # epochs where LR decay occurs
        self.algo.optim_params.policy.regularization.L2 = 0.00          # L2 regularization strength

        self.algo.optim_params.initializer.learning_rate.initial = 1e-4      # policy learning rate
        self.algo.optim_params.initializer.learning_rate.decay_factor = 0.1  # factor to decay LR by (if epoch schedule non-empty)
        self.algo.optim_params.initializer.learning_rate.epoch_schedule = [] # epochs where LR decay occurs
        self.algo.optim_params.initializer.regularization.L2 = 0.00          # L2 regularization strength

        self.algo.optim_params.kwargs.do_not_lock_keys()

        self.algo.max_gradient_norm = None

        # loss weights
        self.algo.loss.l2_weight = 1.0      # L2 loss weight0
        self.algo.loss.l1_weight = 0.0      # L1 loss weight
        self.algo.loss.cos_weight = 0.0     # cosine loss weight

        # MLP network architecture (layers after observation encoder and RNN, if present)1
        self.algo.actor_layer_dims = (1024, 1024)
        self.algo.wm.max_gradient_norm = None
        self.algo.policy.max_gradient_norm = None

        # residual MLP settings
        self.algo.res_mlp.enabled = False
        self.algo.res_mlp.num_blocks = 4
        self.algo.res_mlp.hidden_dim = 1024
        self.algo.res_mlp.use_layer_norm = True

        # stochastic Gaussian policy settings
        self.algo.gaussian.enabled = False              # whether to train a Gaussian policy::
        self.algo.gaussian.fixed_std = False            # whether to train std output or keep it constant
        self.algo.gaussian.init_std = 0.1               # initial standard deviation (or constant)
        self.algo.gaussian.min_std = 0.01               # minimum std output from network
        self.algo.gaussian.std_activation = "softplus"  # activation to use for std output from policy net
        self.algo.gaussian.low_noise_eval = True        # low-std at test-time

        # stochastic GMM policy settings
        self.algo.gmm.enabled = False                   # whether to train a GMM policy
        self.algo.gmm.num_modes = 5                     # number of GMM modes
        self.algo.gmm.min_std = 0.0001                  # minimum std output from network
        self.algo.gmm.std_activation = "softplus"       # activation to use for std output from policy net
        self.algo.gmm.low_noise_eval = True             # low-std at test-time

        # stochastic VAE policy settings
        self.algo.vae.enabled = False                   # whether to train a VAE policy (unused)
        self.algo.vae.method = ""                       # to be specified in json file
        self.algo.vae.latent_dim = 14                   # VAE latent dimnsion - set to twice the dimensionality of action space
        self.algo.vae.latent_clip = None                # clip latent space when decoding (set to None to disable)
        self.algo.vae.kl_weight = 1.                    # beta-VAE weight to scale KL loss relative to reconstruction loss in ELBO
        self.algo.vae.conditioned_on_obs = True

        # VAE decoder settings
        self.algo.vae.decoder.is_conditioned = True                         # whether decoder should condition on observation
        self.algo.vae.decoder.reconstruction_sum_across_elements = False    # sum instead of mean for reconstruction loss

        # VAE prior settings
        self.algo.vae.prior.learn = False                                   # learn Gaussian / GMM prior instead of N(0, 1)
        self.algo.vae.prior.is_conditioned = False                          # whether to condition prior on observations
        self.algo.vae.prior.use_gmm = False                                 # whether to use GMM prior
        self.algo.vae.prior.gmm_num_modes = 10                              # number of GMM modes
        self.algo.vae.prior.gmm_learn_weights = False                       # whether to learn GMM weights
        self.algo.vae.prior.use_categorical = False                         # whether to use categorical prior
        self.algo.vae.prior.categorical_dim = 10                            # the number of categorical classes for each latent dimension
        self.algo.vae.prior.categorical_gumbel_softmax_hard = False         # use hard selection in forward pass
        self.algo.vae.prior.categorical_init_temp = 1.0                     # initial gumbel-softmax temp
        self.algo.vae.prior.categorical_temp_anneal_step = 0.001            # linear temp annealing rate
        self.algo.vae.prior.categorical_min_temp = 0.3                      # lowest gumbel-softmax temp

        self.algo.vae.encoder_layer_dims = (300, 400)                       # encoder MLP layer dimensions
        self.algo.vae.decoder_layer_dims = (300, 400)                       # decoder MLP layer dimensions
        self.algo.vae.prior_layer_dims = (300, 400)                         # prior MLP layer dimensions (if learning conditioned prior)

        self.algo.vae.enc_use_res_mlp = False                              # whether to use residual MLP for encoder
        self.algo.vae.dec_use_res_mlp = False                              # whether to use residual MLP for decoder

        # RNN policy settings
        self.algo.rnn.enabled = False       # whether to train RNN policy
        self.algo.rnn.horizon = 10          # unroll length for RNN - should usually match train.seq_length
        self.algo.rnn.hidden_dim = 400      # hidden dimension size
        self.algo.rnn.rnn_type = "LSTM"     # rnn type - one of "LSTM" or "GRU"
        self.algo.rnn.num_layers = 2        # number of RNN layers that are stacked
        self.algo.rnn.open_loop = False     # if True, action predictions are only based on a single observation (not sequence)
        self.algo.rnn.kwargs.bidirectional = False            # rnn kwargs
        self.algo.rnn.kwargs.do_not_lock_keys()

        # Dreamer
        self.algo.dyn.hidden_dim = 1024
        self.algo.dyn.action_network_dim = 0
        self.algo.dyn.num_layers = 2
        self.algo.dyn.dyn_detach = True
        self.algo.dyn.dyn_weight = 1.0
        self.algo.dyn.use_res_mlp = False # unused
        self.algo.dyn.combine_enabled = True
        self.algo.dyn.smooth_dynamics = False
        self.algo.dyn.smooth_weight = 1.0
        self.algo.dyn.stochastic_inputs = False
        self.algo.dyn.kl_balance = -1
        self.algo.dyn.dyn_class = "deter" # ["deter", "vae"]
        self.algo.dyn.start_training_epoch = None

        # Dreamer reward model
        self.algo.dyn.reward_weight = 1.0
        self.algo.dyn.rew.hidden_dim = 1024
        self.algo.dyn.rew.num_layers = 2
        self.algo.dyn.rew.activation = "leaky_relu"
        self.algo.dyn.rew.rew_class = "binary"
        self.algo.dyn.rew.use_action = True
        self.algo.dyn.rew.use_weighted_loss = False
        self.algo.dyn.rew.use_res_mlp = False

        self.algo.dyn.use_reward = False
        self.algo.dyn.use_dynamics = True
        self.algo.dyn.use_policy = True

        self.algo.dyn.rew.lstm.enabled = False
        self.algo.dyn.rew.lstm.hidden_size = 1024
        self.algo.dyn.rew.lstm.num_layers = 2
        self.algo.dyn.rew.lstm.bidirectional = True
        self.algo.dyn.rew.lstm.seq_length = 10
        self.algo.dyn.rew.lstm.dropout = 0.0
        self.algo.dyn.rew.all_seq_prediction = False

        self.algo.dyn.rew.binary_loss = "bce"
        self.algo.dyn.rew.focal_gamma = 2.0
        self.algo.dyn.rew.focal_alpha = 0.25
        self.algo.dyn.rew.fc_num = 0
    
        self.algo.dyn.load_prev_policy_dyn = False
        self.algo.dyn.load_ckpt = ""
        self.algo.dyn.obs_sg = False

        self.algo.dyn.use_history = True
        self.algo.dyn.use_real = True

        self.algo.dyn.kwargs.do_not_lock_keys()
