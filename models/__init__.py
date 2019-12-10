def create_model(opt):
    if opt.dataset_mode == 'autoencoder':
        from .mesh_autoencoder import AutoEncoderModel
        model = AutoEncoderModel(opt)
    else:
        from .mesh_classifier import ClassifierModel # todo - get rid of this ?
        model = ClassifierModel(opt)
    return model
