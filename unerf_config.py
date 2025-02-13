def config_parser():

    import configargparse

    parser = configargparse.ArgumentParser()
    parser.add_argument("--config", is_config_file=True, help="config file path")
    parser.add_argument("--expname", type=str, help="experiment name")
    parser.add_argument(
        "--basedir", type=str, default="./logs/", help="where to store ckpts and logs"
    )
    parser.add_argument(
        "--datadir",
        type=str,
        default="./data/synthetic_testing/l2",
        help="input data directory",
    )

    # training options
    parser.add_argument("--n_iters", type=int, default=100000)
    parser.add_argument("--ssim_filter_size", type=int, default=7)
    parser.add_argument("--ssim_lambda", type=float, default=0.75)
    parser.add_argument("--loss", type=str, default="l2")
    parser.add_argument("--probe_depth", type=int, default=140)
    parser.add_argument("--probe_width", type=int, default=80)
    parser.add_argument("--output_ch", type=int, default=5)

    parser.add_argument("--tensorboard", action="store_true")
    parser.add_argument("--confmap", type=bool, default=False)
    parser.add_argument("--pose_path", type=str, default=None)

    parser.add_argument(
        "--random_seed", type=int, default=-1
    )  # Set to 0 for deterministic behaviour

    parser.add_argument("--netdepth", type=int, default=8, help="layers in network")
    parser.add_argument("--netwidth", type=int, default=128, help="channels per layer")
    parser.add_argument(
        "--netdepth_fine", type=int, default=8, help="layers in fine network"
    )
    parser.add_argument(
        "--netwidth_fine",
        type=int,
        default=128,
        help="channels per layer in fine network",
    )
    parser.add_argument(
        "--N_rand",
        type=int,
        default=32 * 32 * 4,
        help="batch size (number of random rays per gradient step)",
    )
    parser.add_argument("--lrate", type=float, default=1e-4, help="learning rate")
    parser.add_argument(
        "--lrate_decay",
        type=int,
        default=250,
        help="exponential learning rate decay (in 1000 steps)",
    )
    parser.add_argument(
        "--chunk",
        type=int,
        default=4096 * 16,
        help="number of rays processed in parallel, decrease if running out of memory",
    )
    parser.add_argument(
        "--netchunk",
        type=int,
        default=4096 * 16,
        help="number of pts sent through network in parallel, decrease if running out of memory",
    )

    parser.add_argument(
        "--ft_path",
        type=str,
        default=None,
        help="specific weights npy file to reload for coarse network",
    )

    # rendering options
    parser.add_argument(
        "--N_samples", type=int, default=64, help="number of coarse samples per ray"
    )
    parser.add_argument(
        "--i_embed",
        type=int,
        default=0,
        help="set 0 for default positional encoding, -1 for none",
    )
    parser.add_argument(
        "--i_embed_gauss",
        type=int,
        default=0,
        help="mapping size for Gaussian positional encoding, 0 for none",
    )

    parser.add_argument(
        "--multires",
        type=int,
        default=10,
        help="log2 of max freq for positional encoding (3D location)",
    )
    parser.add_argument(
        "--multires_views",
        type=int,
        default=4,
        help="log2 of max freq for positional encoding (2D direction)",
    )
    parser.add_argument(
        "--raw_noise_std",
        type=float,
        default=0.0,
        help="std dev of noise added to regularize sigma_a output, 1e0 recommended",
    )

    parser.add_argument(
        "--render_only",
        action="store_true",
        help="do not optimize, reload weights and render out render_poses path",
    )
    parser.add_argument(
        "--render_test",
        action="store_true",
        help="render the test set instead of render_poses path",
    )
    parser.add_argument(
        "--render_factor",
        type=int,
        default=0,
        help="downsampling factor to speed up rendering, set 4 or 8 for fast preview",
    )

    # training options

    # dataset options
    parser.add_argument("--dataset_type", type=str, default="us", help="options: us")
    parser.add_argument(
        "--testskip",
        type=int,
        default=8,
        help="will load 1/N images from test/val sets, useful for large datasets like deepvoxels",
    )

    # logging/saving options
    parser.add_argument(
        "--i_print",
        type=int,
        default=1000,
        help="frequency of console printout and metric loggin",
    )
    parser.add_argument(
        "--i_img", type=int, default=1000, help="frequency of tensorboard image logging"
    )
    parser.add_argument(
        "--i_weights", type=int, default=10000, help="frequency of weight ckpt saving"
    )

    parser.add_argument('--reg', action='store_true',
                        help='enables regularization')
    parser.add_argument("--r_tv_penalty", type=float, default=0.00001,
                        help='Weight for TV constrain')
    parser.add_argument("--r_lcc_penalty", type=float, default=0.001,
                        help='Weight for (N)LCC constrain')
    parser.add_argument("--r_clustering", type=float, default=0.,
                        help='Weight for clustering constrain')
    parser.add_argument("--r_clustering_distance", type=float, default=0.,
                        help='Weight for clustering constrain')
    parser.add_argument("--r_max_reflection", type=float, default=0.34,
                        help='Weight for clustering constrain')
    parser.add_argument("--r_warm_up_it", type=int, default=10000,
                        help='Number of iteration for warm_up')
    # segmentation training options
    parser.add_argument('--segm_head', action='store_true',
                        help='enables convex mode')
    parser.add_argument('--segmentation', action='store_true',
                        help='enables convex mode')
    parser.add_argument("--segm_frac", type=int, default=5,
                        help='Number of iteration for warm_up')
    parser.add_argument('--reconstruction', action='store_true',
                        help='enables regularization')
    parser.add_argument('--confidence', action='store_true',
                        help='enables regularization')
    parser.add_argument('--rec_only_theta', action='store_true',
                        help='enables regularization')
    parser.add_argument("--rec_step", type=int, default=20,
                        help='Weight for clustering constrain')
    parser.add_argument("--rec_iter", type=int, default=20000,
                        help='Number of iteration for warm_up')
    parser.add_argument('--rec_only_occ', action='store_true',
                        help='enables regularization')
    return parser
