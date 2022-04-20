import argparse

def get_base_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--scale', type=float, default=0.0000305)
    parser.add_argument('--shift', type=float, default=0.1378)
    parser.add_argument('--image_size', type=int, nargs=2, default=(384, 384))
    parser.add_argument('--model_path', type=str, default='weights/dpt_hybrid-midas-501f0c75.pt')
    parser.add_argument('--dataset_path', type=str, default='data/dpt_hybrid_midas_5k')
    parser.add_argument('--batch_size', type=int, default=16)

    return parser