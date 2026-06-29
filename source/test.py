import math
import sys
from program_options.options_parser import *
from data_loading.data_loading import *
from utilities.utilities import *
from utilities.tester import *
from utilities.loss import YOLOv3Loss
from serialization.serialization import *
import warnings

def main():
    warnings.filterwarnings("once")
    args = parse_program_options(sys.argv[1:])
    main_config = load_main_config(args)

    set_torch_determinism(main_config['general']['determinism'])

    testing_loader = get_coco_data_loader(main_config['dataset']['testing'], main_config['dataset']['image_size'])
    device = torch.device('cpu')

    snapshot_path_base = get_snapshot_path(main_config)

    train_steps_per_epoch = len(testing_loader)
    state = initialize_testing(main_config, device, snapshot_path_base, train_steps_per_epoch)

    tester = Tester(
        model=state['model'],
        device=device,
        main_config=main_config,
        snapshot_path_base=snapshot_path_base,
    )

    with Logger(log_dir=main_config["testing"]["log_dir"]) as logger:
        logger.init_test_bar(train_steps_per_epoch, 0)
        test_result_file = os.path.join(main_config['testing']['save_directory'], os.path.basename(snapshot_path_base) + "_test_results.txt")
        train_stats = tester.run(testing_loader, logger)
        save_metrics_to_txt(train_stats, test_result_file)

if __name__ == '__main__':
    main()