from datasets.ljspeech.dataset import DatasetSource as LJSpeechDatasetSource


def dataset_factory(source, target, hparams):
    if hparams.dataset == "ljspeech.dataset.DatasetSource":
        return LJSpeechDatasetSource(source, target, hparams)
    else:
        raise ValueError("Unkown dataset")


def create_from_tfrecord_files(source_files, target_files, hparams, cycle_length=4,
                               buffer_output_elements=None,
                               prefetch_input_elements=None):
    if hparams.dataset == "ljspeech.dataset.DatasetSource":
        return LJSpeechDatasetSource.create_from_tfrecord_files(source_files, target_files, hparams,
                                                                cycle_length=cycle_length,
                                                                buffer_output_elements=buffer_output_elements,
                                                                prefetch_input_elements=prefetch_input_elements)
    else:
        raise ValueError("Unkown dataset")
