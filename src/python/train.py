def setup_data_loader() -> str:
    from fastai.vision import ImageDataBunch
    import dill
    import codecs
    bunch = ImageDataBunch.from_folder(
        "test/images", train="training", valid="test",
        size=112)
    return codecs.encode(dill.dumps(bunch), "base64").decode()


def fit_squeezenet(loader: str, storage: str, pretrained: bool = False,
                   name: str = "squeezenet.model") -> str:
    import dill
    from fastai.vision import cnn_learner, accuracy, models
    import codecs

    loader = dill.loads(codecs.decode(loader.encode(), "base64"))
    learner = cnn_learner(
        loader, models.squeezenet1_0, pretrained=False, metrics=accuracy,
        silent=False, add_time=True)
    learner.fit_one_cycle(1)

    outpath = storage + "/" + name
    with open(outpath, "wb") as fp:
        dill.dump(learner, fp)

    return outpath

def fit_resnet18(loader: str, storage: str, pretrained: bool = False,
              name: str = "res18.model") -> str:
    import dill
    from fastai.vision import cnn_learner, accuracy, models
    import codecs

    loader = dill.loads(codecs.decode(loader.encode(), "base64"))
    learner = cnn_learner(
        loader, models.resnet18, pretrained=False, metrics=accuracy)
    learner.fit_one_cycle(1)

    outpath = storage + "/" + name
    with open(outpath, "wb") as fp:
        dill.dump(learner, fp)

    return outpath


def fit_resnet(loader: str, storage: str, pretrained: bool = False,
               name: str = "resnet.model") -> str:
    import dill
    from fastai.vision import cnn_learner, accuracy, models
    import codecs

    loader = dill.loads(codecs.decode(loader.encode(), "base64"))

    learner = cnn_learner(
        loader, models.resnet34, pretrained=False, metrics=accuracy)
    learner.fit_one_cycle(1)

    outpath = storage + "/" + name
    with open(outpath, "wb") as fp:
        dill.dump(learner, fp)

    return outpath


def get_accuracy(learner: str) -> float:
    from fastai.vision import accuracy
    import dill

    with open(learner, 'rb') as fp:
        learner = dill.load(fp)

    return float(accuracy(*learner.get_preds()).numpy())


def get_confusion(learner: str) -> str:
    from fastai.vision import ClassificationInterpretation
    import dill

    with open(learner, 'rb') as fp:
        learner = dill.load(fp)

    interpreter = ClassificationInterpretation.from_learner(learner)
    return str(interpreter.confusion_matrix())

def main():
    import os
    path = "out"

    if not os.path.isdir(path):
        os.makedirs(path)

    loader = setup_data_loader()
    squeezenet = fit_squeezenet(loader, path)
    resnet = fit_resnet(loader, path)
    res18 = fit_resnet18(loader, path)
    nets = [squeezenet, resnet, res18]
    accuracy = [get_accuracy(net) for net in nets]
    confusions = [get_confusion(net) for net in nets]


if __name__ == "__main__":
    main()
