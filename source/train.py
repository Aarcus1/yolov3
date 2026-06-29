import sys
from program_options.options_parser import *
from data_loading.data_loading import *
from utilities.utilities import *
from utilities.trainer import *
from utilities.loss import YOLOv3Loss
from serialization.serialization import *
import warnings

def main():
    warnings.filterwarnings("once")  # to avoid flooding the console output
    args = parse_program_options(sys.argv[1:])
    main_config = load_main_config(args)

    set_torch_determinism(main_config['general']['determinism'])

    training_loader = get_coco_data_loader(main_config['dataset']['training'], main_config['dataset']['image_size'])
    validation_loader = get_coco_data_loader(main_config['dataset']['validation'], main_config['dataset']['image_size'])

    device = get_device(main_config['training']['device'])

    loss_config = main_config['training']['loss']
    loss_fn = YOLOv3Loss(
        num_classes=main_config['dataset']['class_count'],
        lambda_coord=float(loss_config['lambda_coord']),
        lambda_obj=float(loss_config['lambda_obj']),
        lambda_noobj=float(loss_config['lambda_noobj']),
        lambda_cls=float(loss_config['lambda_cls']),
    )

    snapshot_path_base = get_snapshot_path(main_config)

    train_steps_per_epoch = len(training_loader)
    state = initialize_training(main_config, device, snapshot_path_base, train_steps_per_epoch)

    trainer = Trainer(
        model=state['model'],
        optimizer=state['optimizer'],
        scaler=state['scaler'],
        scheduler=state['scheduler'],
        loss_fn=loss_fn,
        device=device,
        main_config=main_config,
        snapshot_path_base=snapshot_path_base,
    )

    start_epoch = state["start_epoch"]
    num_epochs = main_config['training']['epochs']

    with Logger(log_dir=main_config["training"]["log_dir"]) as logger:
        trainer.train(
            training_loader=training_loader,
            validation_loader=validation_loader,
            logger=logger,
            main_config=main_config,
            max_epoch=num_epochs,
            start_epoch=start_epoch,
        )

if __name__ == '__main__':
    main()
